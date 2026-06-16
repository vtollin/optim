#pragma once
#include <Eigen/Dense>
#include <string>

namespace optim {
struct OptimizationResult {
    Eigen::VectorXd x_opt;
    double f_val;
    int iterations;
    bool converged;
    std::string message;
};
} // namespace optim