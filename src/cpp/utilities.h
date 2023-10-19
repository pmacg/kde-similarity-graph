#ifndef SIMILARITY_GRAPH_UTILITIES_H
#define SIMILARITY_GRAPH_UTILITIES_H

#include <Eigen/Dense>
#include <vector>
#include <graph.h>

/**
 * Create a matrix of points sampled uniformly at random from the unit cube.
 *
 * @param numPoints the number of points to generate
 * @param numDimensions the dimension of the cube from which to sample
 * @return
 */
Eigen::MatrixXd generateRandomPoints(int numPoints, int numDimensions);

Eigen::MatrixXd readDataFromFile(const std::string& filename);

/**
 * Given an n by d matrix, compute the degrees of the similarity graph by the
 * naive n**2 method.
 * @param points the Eigen matrix of the data points
 * @param sigma the parameter of the gaussian kernel
 * @return a vector of the vertex degrees
 */
std::vector<double> computeDegreesNaively(const Eigen::MatrixXd& points, double sigma);

SprsMat adjacencyToLaplacian(SprsMat& adj_mat);

#endif //SIMILARITY_GRAPH_UTILITIES_H
