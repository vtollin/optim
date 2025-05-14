#pragma once
#include <Eigen/Dense>
#include <string>

struct OptimizationResult {
    Eigen::VectorXd xOpt;
    double fVal;
    int iterations;
    bool converged;
    std::string message; 
};