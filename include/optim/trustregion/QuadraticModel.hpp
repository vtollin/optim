#pragma once
#include <Eigen/Dense>

namespace optim::trustregion {
struct QuadraticModel {
    double f_x;
    Eigen::VectorXd g;
    Eigen::MatrixXd B;
};
} // namespace optim::trustregion
