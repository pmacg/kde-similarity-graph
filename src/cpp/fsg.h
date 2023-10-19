#ifndef SIMILARITY_GRAPH_FSG_H
#define SIMILARITY_GRAPH_FSG_H

#include <Eigen/Dense>
#include <vector>
#include "graph.h"
#include "kde.h"
#include <iostream>

// Define some debugging messages
#ifdef DEBUG
#define DEBUG_MSG(fmt, ...) { fprintf(stderr, fmt, __VA_ARGS__); }
#else
#define DEBUG_MSG(fmt, ...) { }
#endif
#ifdef TRACE
#define DEBUG_MSG(fmt, ...) { fprintf(stderr, fmt, __VA_ARGS__); }
#define TRACE_MSG(fmt, ...) { fprintf(stderr, fmt, __VA_ARGS__); }
#else
#define TRACE_MSG(fmt, ...) { }
#endif

// Define the structs which will be used in the algorithm.
struct Interval;
struct Sample;

/**
 * Construct a similarity graph from the given points, using the fast similarity
 * graph algorithm.
 *
 * @param points the data points from which to construct the similarity graph
 * @param sigma the sigma parameter of the gaussian kernel
 * @return a sparse Eigen matrix giving the similarity matrix
 */
SprsMat fast_similarity_graph(Eigen::MatrixXd& points, double sigma,
                              KDESolver& solver);
SprsMat fast_similarity_graph_debug(Eigen::MatrixXd& points, double sigma,
                                    KDESolver& solver);

#endif //SIMILARITY_GRAPH_FSG_H
