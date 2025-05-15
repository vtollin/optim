#pragma once
#include "OptimizerBase.hpp"
#include "LineSearch.hpp"
#include "Eigen/Dense"

class SteepestDescent : public OptimizerBase {
public:
    SteepestDescent(
        std::shared_ptr<LineSearch> lineSearchStrategy, 
        int max_iterations_ = 10000,
        double tol_ = 1e-6);

    OptimizationResult optimize(
        ObjectiveFunction& f,
        const Eigen::VectorXd& x0) override; 
private: 
    std::shared_ptr<LineSearch> line_search_; 
    int max_iterations_;
    double tol_;
};
