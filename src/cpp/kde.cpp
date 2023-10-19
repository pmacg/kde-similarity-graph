#include "kde.h"
#include <cmath>
#include <random>
#include <functional>
#include <thread>
#include <future>
#include <iostream>
#include "fgt.hpp"

const double PI = 3.1415926535;
const double INV_SQRT_2PI = 1.0 / std::sqrt(2 * PI);

//--------------------------------------------------------
// Improved Fast Gauss Transform KDE
//--------------------------------------------------------
IFGT::IFGT() {}

Eigen::VectorXd IFGT::query(const Eigen::MatrixXd& data,
                            const Eigen::MatrixXd& query,
                            double bandwidth,
                            bool debug) {
    // Call through to the IFGT code - the bandwidth must be at most 1!
    return fgt::ifgt(data, query, bandwidth, 1.0);
}
