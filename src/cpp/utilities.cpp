#include "utilities.h"
#include <random>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <graph.h>

Eigen::MatrixXd generateRandomPoints(int numPoints, int numDimensions) {
  // Create a random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(0, 1.0);

  // Create an Eigen matrix to hold the points
  Eigen::MatrixXd points(numPoints, numDimensions);

  // Generate the points
  for (int i = 0; i < numPoints; i++) {
    // Generate random numbers for the current point
    for (int j = 0; j < numDimensions; j++) {
      points(i, j) = dist(gen);
    }
  }

  return points;
}

Eigen::MatrixXd readDataFromFile(const std::string &fileName) {
    int rows = 0; // variable to store the number of rows
    int cols = 0; // variable to store the number of columns
    std::string line; // variable to store the current line being read
    std::ifstream file(fileName); // create an input file stream

    // read the file line by line
    while (std::getline(file, line)) {
        if (rows == 0) {
            std::stringstream lineStream(line);
            std::string cell;
            // count the number of columns by counting the number of commas in the first line
            while (std::getline(lineStream, cell, ',')) {
                cols++;
            }
        }
        // increment the number of rows
        rows++;
    }
    file.close();
    file.open(fileName);
    Eigen::MatrixXd data(rows, cols);
    int row = 0;
    // read the file again to store the contents in the matrix
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        int col = 0;
        while (std::getline(lineStream, cell, ',')) {
            data(row, col) = std::stod(cell);
            col++;
        }
        row++;
    }
    return data;
}


std::vector<double> computeDegreesNaively(const Eigen::MatrixXd& points, double sigma) {
  // Get the number of points
  int numPoints = points.rows();

  // Create a vector to hold the results
  std::vector<double> results(numPoints);

  // Compute the sum of the Gaussian kernel function for each point
  for (int i = 0; i < numPoints; i++) {
    // Compute the sum of the Gaussian kernel function for the current point
    double sum = 0;
    for (int j = 0; j < numPoints; j++) {
      if (i == j) continue; // Skip the current point itself

      // Compute the distance between the current point and the other point
      double distance = 0;
      for (int k = 0; k < points.cols(); k++) {
        distance += std::pow(points(i, k) - points(j, k), 2);
      }

      // Add the Gaussian kernel function applied to the distance to the sum
      sum += std::exp(-distance / (std::pow(sigma, 2)));
    }

    // Store the result for the current point
    results[i] = sum;
  }

  return results;
}

SprsMat adjacencyToLaplacian(SprsMat& adj_mat) {
  stag::Graph g(adj_mat);
  return *g.normalised_laplacian();
}
