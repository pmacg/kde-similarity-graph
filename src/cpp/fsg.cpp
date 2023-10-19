#include "fsg.h"
#include "graph.h"
#include <cmath>
#include "fgt.hpp"
#include <random>
#include <deque>
#include <queue>
#include <future>
#include <thread>
#include <chrono>
#include <set>
#include "cluster.h"
#include "kde.h"

// Define the constant interval length at which we process the interval with
// brute force.
static long INTERVAL_MIN_LENGTH = 1000;

// Define the number of worker threads;
static long WORKER_THREADS = 30;

// Define a struct describing some 'interval' of the indices of the data points.
// We actually define this as just a vector of indices for the points.
// This means that an 'interval' need not be a consecutive list of points
struct Interval {
  std::vector<long> point_idxs;
  std::vector<Sample*> samples;

  Interval(std::vector<long> pidxs, std::vector<Sample*> smpls)
    : point_idxs(pidxs)
    , samples(smpls)
    {}
};

// Define a struct to hold the fields corresponding to one of the samples
// for the fast similarity graph algorithm.
struct Sample {
  long originating_vertex;
  long other_vertex;
  bool determined;
  double random_sample;
};

bool sample_sorter(Sample const& lhs, Sample const& rhs) {
  return lhs.random_sample < rhs.random_sample;
}

Interval fsg_generate_samples(Eigen::MatrixXd& points,
                              double sigma,
                              long num_samples,
                              Sample* samples_array) {
  long n = points.rows();
  std::vector<Sample*> samples;

  // Create a random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  // For each point in the list of points, we sample num_samples samples.
  // If num_samples is O(log(n)), then this entire iteration takes
  // O(n log(n))
  long sample_idx = 0;
  for (long i = 0; i < n; i++) {
    for (long j = 0; j < num_samples; j++) {
      // Generate a random sample between 0 and 1
      double r = dist(gen);

      // Add the sample object to the samples vector.
      TRACE_MSG("Creating sample: {%ld, %f}\n",
                i, r)

      samples_array[sample_idx] = {i, i, false, r};
      samples.push_back(samples_array + sample_idx);
      sample_idx++;
    }
  }

  DEBUG_MSG("Total samples: %ld\n", samples.size());

  // Initialise the interval object
  std::vector<long> points_idxs;
  for (long i = 0; i < n; i++) {
    points_idxs.push_back(i);
  }
  Interval new_interval = {points_idxs, samples};
  return new_interval;
}

void fsg_add_interval_back(std::vector<long>& int_points,
                           std::vector<Sample*>& smpls,
                           std::deque<Interval>* queue,
                           std::mutex* q_mutex) {
  std::unique_lock<std::mutex> lock(*q_mutex);
  queue->emplace_back(int_points, smpls);
}

void fsg_add_interval_front(std::vector<long>& int_points,
                            std::vector<Sample*>& smpls,
                            std::deque<Interval>* queue,
                            std::mutex* q_mutex) {
  std::unique_lock<std::mutex> lock(*q_mutex);
  queue->emplace_front(int_points, smpls);
}

bool fsg_interval_q_empty(std::deque<Interval>* queue,
                          std::mutex& q_mutex) {
  std::unique_lock<std::mutex> lock(q_mutex);
  return queue->empty();
}

/**
 * Compute the exact weight between the points at indices v1 and v2.
 *
 * @param points
 * @param v1
 * @param v2
 * @return
 */
double fsg_compute_weight(const Eigen::MatrixXd& points, long v1, long v2, double sigma) {
  double distance = 0;
  for (int k = 0; k < points.cols(); k++) {
    distance += std::pow(
        points(v1, k) - points(v2, k),
        2);
  }
  return std::exp(-distance / (std::pow(sigma, 2)));
}

/**
 * Process an interval by the brute force method.
 *
 * @param points
 * @param sigma
 * @param intrvl
 */
void fsg_processs_interval_brute_force(const Eigen::MatrixXd& points,
                                       const double sigma,
                                       const Interval& intrvl) {
  // Iterate through each sample in the interval and select their accompanying
  // data point.
  for (Sample* smpl : intrvl.samples) {
    long this_vertex = smpl->originating_vertex;
    double degree = 0;
    std::queue<double> weights;

    // Compute the total degree of the sample source point to the
    // vertices in the interval.
    for (long i : intrvl.point_idxs) {
      // Compute the weight to point i
      double weight = fsg_compute_weight(points, this_vertex, i, sigma);
      degree += weight;
      weights.push(weight);
    }

    // Now we know the true degree, figure out which of the vertices in the
    // interval should be returned.
    double degree_so_far = 0;
    for (long i : intrvl.point_idxs) {
      degree_so_far += weights.front() / degree;
      weights.pop();

      if (degree_so_far >= smpl->random_sample) {
        // We've found our second vertex for the sample!
        smpl->other_vertex = i;
        smpl->determined = true;
      }

      if (smpl->determined) {
        break;
      }
    }

    // If for some reason we haven't found the other vertex, set it to
    // the last element in the current interval, in case it's caused by
    // floating point issues.
    if (!smpl->determined) {
      DEBUG_MSG("Fudge - setting other vertex to %ld\n", intrvl.point_idxs.back())
      smpl->other_vertex = intrvl.point_idxs.back();
      smpl->determined = true;
    }
  }
}

std::vector<long> fsg_get_unique_points_in_samples(std::vector<Sample*>& samples) {
  std::vector<long> points;
  long last_point = -1;
  for (Sample* smpl : samples) {
    if (smpl->originating_vertex != last_point) {
      // We make the assumption that the sample points are ordered according
      // to the originating vertex.
      assert(smpl->originating_vertex > last_point);
      points.push_back(smpl->originating_vertex);
      last_point = smpl->originating_vertex;
    }
  }

  // Convert the set to a vector
  return points;
}

void fsg_check_interval(Interval& intrvl) {
  // Check that every sample has a valid random sample double
  for (Sample* smpl : intrvl.samples) {
    assert(!std::isnan(smpl->random_sample));
  }
}

/**
 * Process the first interval on the queue. If the interval has more than
 * a constant number of elements, then this will add two sub-intervals
 * to the back of the queue to be processed.
 *
 * The running time of the algorithm to process a single interval is O(M + N)
 * where M is the size of the interval and N is the number of samples in the
 * interval.
 *
 * @param points
 * @param sigma the sigma parameter of the gaussian kernel
 * @param int_queue
 * @param gen the random number generator to use
 */
int fsg_process_interval(const Eigen::MatrixXd& points,
                         const double& sigma,
                         Interval intrvl,
                         std::deque<Interval>* int_queue,
                         KDESolver* solver,
                         std::mutex* q_mutex,
                         const bool& debug) {
  // Get the unique points in the samples on this interval
  std::vector<long> unique_points = fsg_get_unique_points_in_samples(intrvl.samples);

  DEBUG_MSG("Processing interval with %ld points, and %ld samples from %ld unique points\n",
            intrvl.point_idxs.size(),
            intrvl.samples.size(),
            unique_points.size())

  // Check the interval
  fsg_check_interval(intrvl);

  // Create a random number generator
  std::random_device rd;
  std::mt19937 gen(rd());

  // If the length of the interval is less than a constant, then we process it exactly
  // and do not add any new intervals to the queue.
  long int_length = intrvl.point_idxs.size();
  if (int_length <= INTERVAL_MIN_LENGTH) {
    DEBUG_MSG("Interval has length %ld - finishing with brute force\n",
              int_length)

    fsg_processs_interval_brute_force(points, sigma, intrvl);

    // Return from the function - we have set the second vertex for all of the
    // samples in this interval.
    return 0;
  }

  // This interval is too large to be brute forced - we will split the interval
  // into two pieces and assign the samples to one of the new intervals, then
  // add the intervals back to the queue to be processed.
  //
  // Note that this will always split the size of the interval in two, and so
  // we are guaranteed to terminate since the size always decreases and so will
  // be solved by the brute force algorithm at some point.

  // Start by creating the two new sample vectors
  std::vector<Sample*> int1_samples;
  std::vector<Sample*> int2_samples;

  // Split the points in the current interval into the next two intervals
  // Repeat the random split until we have points in both sub-intervals
  std::vector<long> int1_point_idxs;
  std::vector<long> int2_point_idxs;
  std::uniform_int_distribution<long> rand_dim(0, points.cols() - 1);
  std::uniform_int_distribution<long> rand_interval_point(
      0, (long) intrvl.point_idxs.size() - 1);

  // Perform the split. If the number of points is greater than 2000,
  // always split in half.
  if (intrvl.point_idxs.size() > 0){
    long split_idx = intrvl.point_idxs.size() / 2;
    int1_point_idxs.assign(intrvl.point_idxs.begin(), intrvl.point_idxs.begin() + split_idx);
    int2_point_idxs.assign(intrvl.point_idxs.begin() + split_idx, intrvl.point_idxs.end());
  } else {
    // Otherwise, we split at random on some dimension
    long split_dimension = rand_dim(gen);
    DEBUG_MSG("Spliting dimension %ld\n", split_dimension);

    // Find the mid-point of this dimension

    long split_point_idx = rand_interval_point(gen);
    double split_point = points(intrvl.point_idxs.at(split_point_idx), split_dimension);
    for (long point_idx : intrvl.point_idxs) {
      if (points(point_idx, split_dimension) <= split_point) {
        int1_point_idxs.push_back(point_idx);
      } else {
        int2_point_idxs.push_back(point_idx);
      }
    }

    // Check whether this is a good split
    if (int1_point_idxs.empty() || int2_point_idxs.empty()) {
      DEBUG_MSG("Bad split. Splitting in half.\n", 0)

      // This is not a good split. Split the points in half instead
      int1_point_idxs.clear();
      int2_point_idxs.clear();

      long split_idx = intrvl.point_idxs.size() / 2;
      int1_point_idxs.assign(intrvl.point_idxs.begin(), intrvl.point_idxs.begin() + split_idx);
      int2_point_idxs.assign(intrvl.point_idxs.begin() + split_idx, intrvl.point_idxs.end());
    }
  }

  // Create three new matrices for use with the Fast Gaussian Transform function
  // The first contains only the points corresponding to the samples in the
  // current interval
  Eigen::MatrixXd sample_points(unique_points.size(), points.cols());
  long idx = 0;
  for (long v1 : unique_points) {
    for (long k = 0; k < points.cols(); k++) {
      sample_points(idx, k) = points(v1, k);
    }
    idx++;
  }

  // The next is the matrix of all points in the new int1 interval
  Eigen::MatrixXd int1_points(int1_point_idxs.size(), points.cols());
  idx = 0;
  for (long point_idx : int1_point_idxs) {
    for (long k = 0; k < points.cols(); k++) {
      int1_points(idx, k) = points(point_idx, k);
    }
    idx++;
  }

  // And finally, get the points on the new int2 interval
  Eigen::MatrixXd int2_points(int2_point_idxs.size(), points.cols());
  idx = 0;
  for (long point_idx : int2_point_idxs) {
    for (long k = 0; k < points.cols(); k++) {
      int2_points(idx, k) = points(point_idx, k);
    }
    idx++;
  }
  assert(int1_points.rows() + int2_points.rows() == intrvl.point_idxs.size());

  // Now, we compute the approximate degrees for each of the sample points
  // for each of the sub-intervals.
  auto task1 = [&solver, &int1_points, &sample_points, &sigma, &debug] ()
  {
    return solver->query(int1_points, sample_points, sigma, debug);
  };
  std::future<Eigen::VectorXd> future1 = std::async(std::launch::async,
                                                        task1);

  auto task2 = [&solver, &int2_points, &sample_points, &sigma, &debug] ()
  {
    return solver->query(int2_points, sample_points, sigma, debug);
  };
  std::future<Eigen::VectorXd> future2 = std::async(std::launch::async,
                                                        task2);

  Eigen::VectorXd int1_degrees = future1.get();
  assert(int1_degrees.rows() == unique_points.size());

  Eigen::VectorXd int2_degrees = future2.get();
  assert(int2_degrees.rows() == unique_points.size());

  // Now, iterate through the sample points and assign them to either
  // interval 1 or interval 2.
  long unique_pt_index = 0;
  for (Sample* smpl : intrvl.samples) {
    // Check whether we need to move onto the next unique point index
    if (smpl->originating_vertex != unique_points.at(unique_pt_index)) {
      unique_pt_index++;
      assert(smpl->originating_vertex == unique_points.at(unique_pt_index));
    }

    // We do not correct the calculated degrees - we are operating as if there
    // is a self-loop on every vertex.
    double int1_deg = int1_degrees(unique_pt_index, 0);
    double int2_deg = int2_degrees(unique_pt_index, 0);

    double total_deg = int1_deg + int2_deg;
    double int1_deg_prop = int1_deg / total_deg;

    // Check for total_degree being 0. In this case, we say that int1_deg_prop
    // is 1/2.
    if (total_deg == 0) {
      int1_deg_prop = 0.5;
    }
    assert(!std::isnan(int1_deg_prop));

    if (smpl->random_sample <= int1_deg_prop) {
      // This sample is going in interval 1
      smpl->random_sample = smpl->random_sample / int1_deg_prop;
      int1_samples.push_back(smpl);
      assert(!std::isnan(smpl->random_sample));
    } else {
      // This sample is going in interval 2
      smpl->random_sample = (smpl->random_sample - int1_deg_prop) /
          (1 - int1_deg_prop);
      int2_samples.push_back(smpl);
      assert(!std::isnan(smpl->random_sample));
    }
  }

  // Add the two new intervals to the front of the queue - this results
  // in a depth-first traversal of the tree which reduces the memory requirement
  fsg_add_interval_back(int2_point_idxs, int2_samples, int_queue, q_mutex);
  fsg_add_interval_back(int1_point_idxs, int1_samples, int_queue, q_mutex);

  return 0;
}

SprsMat construct_adj_mat_from_sampled_edges(Eigen::MatrixXd& points,
                                             Sample* samples,
                                             double sigma,
                                             stag_int num_vertices,
                                             stag_int num_samples) {
  // iterate through the samples and construct the similarity matrix.
  SprsMat adj_mat(num_vertices, num_vertices);
  std::vector<EdgeTriplet> non_zero_entries;
  for (long sample_idx = 0; sample_idx < num_samples; sample_idx++) {
    Sample* smpl = &samples[sample_idx];
    assert(smpl->determined);

    long i = smpl->originating_vertex;
    long j = smpl->other_vertex;
    double w = fsg_compute_weight(points, i, j, sigma);

    assert(!std::isnan(w));

    // Add the edge to the adjacency matrix
    non_zero_entries.emplace_back(i, j, w);
    if (i != j) {
      non_zero_entries.emplace_back(j, i, w);
    }
  }
  adj_mat.setFromTriplets(non_zero_entries.begin(), non_zero_entries.end());

  return adj_mat;
}

SprsMat fast_similarity_graph_internal(Eigen::MatrixXd& points, double sigma,
                              KDESolver& solver, bool debug) {
  // Get the number of data points
  long n = points.rows();
  long logn = (long) log((double) n);
  assert(logn > 0);
  DEBUG_MSG("Log(n): %ld\n", logn)

  // Begin by constructing the samples for each data point.
  // We will sample 3 log(n) samples.
  long num_samples = 3 * logn;
  long total_samples = n * num_samples;
  DEBUG_MSG("Number of samples per point: %ld\n", num_samples)
  DEBUG_MSG("Total number of samples: %ld\n", total_samples)

  // Initialise the memory in which to store the samples
  auto* all_samples = new Sample[total_samples];

  // Create a random number generator for the duration of the algorithm
  Interval first_interval = fsg_generate_samples(points,
                                                 sigma,
                                                 num_samples,
                                                 all_samples);

  // Create a queue of intervals to be processed.
  std::deque<Interval> interval_queue;
  std::mutex interval_queue_mutex;
  interval_queue.push_back(first_interval);

  // And a queue of worker threads which are processing intervals.
  std::vector<std::future<int>> worker_queue;

  // Process the intervals on the queue until none remain.
  while (!fsg_interval_q_empty(&interval_queue, interval_queue_mutex) ||
          !worker_queue.empty()) {
    if (fsg_interval_q_empty(&interval_queue, interval_queue_mutex) ||
        worker_queue.size() == WORKER_THREADS) {
      // Check for any workers which are ready. This is a hack and will spin up a full cpu.
      bool worker_ready = false;
      int worker_idx = 0;
      if (worker_queue.size() == 1) {
        // If the worker queue has only one element, then, we can just
        // wait for it directly.
        worker_ready = true;
      }
      while (!worker_ready) {
        for (int i = 0; i < worker_queue.size(); i++) {
          if (worker_queue.at(i).wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            // This worker is ready
            worker_ready = true;
            worker_idx = i;
          }
        }
      }

      // Get the worker which is ready
      std::future<int> next_future = std::move(worker_queue.at(worker_idx));
      worker_queue.erase(worker_queue.begin() + worker_idx);
      int result = next_future.get();

      // The result should always be success
      assert(result == 0);
    } else {
      // There is at least one interval on the queue to be processed, and
      // the worker queue is not full. Process the next interval.

      // Pop the front of the interval queue
      std::unique_lock<std::mutex> lock(interval_queue_mutex);
      Interval intrvl = interval_queue.front();
      interval_queue.pop_front();
      lock.unlock();

      // Start the interval processing asynchronously
      // Note - processing an interval also deletes its data and frees the
      // memory for the Interval and samples vector.
      auto task = [&points, &sigma, intrvl, &interval_queue, &solver,
                   &interval_queue_mutex, &debug] ()
      {
        return fsg_process_interval(points,
                                    sigma,
                                    intrvl,
                                    &interval_queue,
                                    &solver,
                                    &interval_queue_mutex,
                                    debug);
      };

      std::future<int> new_future = std::async(std::launch::async,
                                               task);
      worker_queue.push_back(std::move(new_future));
    }
  }

  // Now, iterate through the samples and construct the similarity matrix.
  SprsMat adj_mat = construct_adj_mat_from_sampled_edges(points,
                                                         all_samples,
                                                         sigma,
                                                         n,
                                                         total_samples);

  // Free the samples memory
  delete [] all_samples;

  DEBUG_MSG("Completed FSG on %ld datapoints\n", n)

  return adj_mat;
}

SprsMat fast_similarity_graph(Eigen::MatrixXd& points, double sigma,
                              KDESolver& solver) {
  return fast_similarity_graph_internal(points, sigma, solver, false);
}
SprsMat fast_similarity_graph_debug(Eigen::MatrixXd& points, double sigma,
                                    KDESolver& solver) {
  return fast_similarity_graph_internal(points, sigma, solver, true);
}
