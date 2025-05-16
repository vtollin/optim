#pragma once
#include <Eigen/Dense>
#include <string>

struct OptimizationResult {
    Eigen::VectorXd x_opt;
    double f_val;
    int iterations;
    bool converged;
    std::string message;
};