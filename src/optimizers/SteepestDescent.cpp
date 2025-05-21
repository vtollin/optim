#include "optimization/Optimizer/SteepestDescent.hpp"
#include "optimization/LineSearch/SearchStrategyBase.hpp"
#include "optimization/Logger.hpp"
#include "optimization/ObjectiveFunctionBase.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <string>

using namespace Optimizer;

SteepestDescent::SteepestDescent(std::shared_ptr<LineSearch::SearchStrategyBase> search_strategy,
                                 int max_iterations, double tol, std::shared_ptr<Logger> logger)
    : search_strategy_(search_strategy)
    , max_iterations_(max_iterations)
    , tol_(tol)
    , logger_(logger) {
}

OptimizationResult SteepestDescent::optimize(const ObjectiveFunctionBase &f,
                                             const Eigen::VectorXd &x0) {
    int k = 0;
    bool converged = false;
    Eigen::VectorXd x = x0;
    std::string msg = "Failed to converge in specified iterations.";

    if (f.sourceDimension() != x.size()) {
        throw std::invalid_argument("Initial vector is not in the source of objective function.");
    }

    for (; k < max_iterations_; ++k) {
        Eigen::VectorXd grad = f.gradient(x);
        if (grad.norm() < tol_) {
            converged = true;
            msg = "Converged successfully";
            break;
        }
        Eigen::VectorXd direction = -grad;
        Eigen::VectorXd step = search_strategy_->computeStep(f, x, direction, grad);
        if (step.norm() == 0.0) {
            converged = false;
            msg = "Line search stalled";
            break;
        }
        if (logger_) {
            logger_->logIteration(k, x, grad, f.evaluate(x), step);
        }
        x += step;
    }
    return OptimizationResult{x, f.evaluate(x), k, converged, msg};
}