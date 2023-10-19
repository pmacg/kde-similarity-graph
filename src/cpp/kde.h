//
// Methods for Kernel Density Estimation
//
#ifndef SIMILARITY_GRAPH_KDE_H
#define SIMILARITY_GRAPH_KDE_H

#include <Eigen/Dense>
#include <memory>

/**
 * Abstract class providing methods for solving the Kernel Density Estimation problem.
 */
class KDESolver {
public:
    // Query method
    virtual Eigen::VectorXd query(const Eigen::MatrixXd& data,
                                  const Eigen::MatrixXd& query,
                                  double bandwidth,
                                  bool debug) = 0;

    // Destructor
    virtual ~KDESolver() = default;
};

/**
 * KDE based on the Improved Fast Gauss Transform.
 */
class IFGT : public KDESolver {
public:
    // Constructor
    IFGT();

    // Query method
    Eigen::VectorXd query(const Eigen::MatrixXd& data,
                          const Eigen::MatrixXd& query,
                          double bandwidth,
                          bool debug) override;

    // Destructor
    ~IFGT() override = default;
};

#endif //SIMILARITY_GRAPH_KDE_H
