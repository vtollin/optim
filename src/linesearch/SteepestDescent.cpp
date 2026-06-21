#include "optim/linesearch/SteepestDescent.hpp"
#include "optim/linesearch/StepLengthPolicy.hpp"
#include "optim/logger/Logger.hpp"
#include "optim/Functions.hpp"
#include "optim/OptimizerUtility.hpp"
#include <Eigen/Dense>
#include <string>

using namespace optim::linesearch;
using optim::OptimizationResult;
using optim::DifferentiableFunction;

SteepestDescent::SteepestDescent(SearchStrategy search_strategy, int max_iterations,
                                 optim::ConvergenceCriteria criteria,
                                 std::shared_ptr<optim::logger::Logger> logger)
    : LineSearchBase(search_strategy, max_iterations, criteria, logger) {
    if (max_iterations_ < 1) {
        throw std::invalid_argument("[SteepestDescent] Max iterations must be positive.");
    }
    if (criteria.grad_tol <= 0 || criteria.f_tol <= 0 || criteria.step_tol <= 0) {
        throw std::invalid_argument("[SteepestDescent] Tolerances must be positive.");
    }
}

// Iterates from x0 using the negative gradient as the search direction, delegating step
// length selection to the configured line search strategy. N&W Section 3.1, p. 30.
OptimizationResult SteepestDescent::optimize(const DifferentiableFunction &f,
                                             const Eigen::VectorXd &x0) {
    if (x0.size() != f.sourceDimension()) {
        throw std::invalid_argument(
            "[SteepestDescent] Initial vector is not in the source of objective function.");
    }

    Eigen::VectorXd x = x0;
    bool converged = false;
    std::string msg = "Failed to converge in specified iterations.";
    int k = 0;
    for (; k < max_iterations_; ++k) {
        Eigen::VectorXd grad = f.gradient(x);
        if (grad.norm() < criteria_.grad_tol * (1.0 + x.norm())) { // relative tolerance with absolute floor
            converged = true;
            msg = "Converged: gradient norm fell below tolerance.";
            break;
        }
        Eigen::VectorXd direction = -grad;
        double alpha = step_length_policy_->computeStep(f, x, direction, grad);
        if (alpha == 0.0) {
            msg = "Search strategy returned 0 step.";
            break;
        }
        Eigen::VectorXd step = alpha * direction;
        double f0 = f.evaluate(x);
        if (logger_ && logger_->shouldLog(optim::logger::Verbosity::INFO)) {
            logger_->logIteration(optim::logger::IterationInfo{k, x, grad, step, f0});
        }
        x += step;
        if (step.norm() < criteria_.step_tol * (1.0 + x.norm())) {
            converged = true;
            msg = "Converged: step size fell below tolerance.";
            break;
        }
        double f1 = f.evaluate(x);
        if (std::abs(f1 - f0) < criteria_.f_tol * (std::abs(f0) + 1.0)) {
            converged = true;
            msg = "Converged: objective change fell below tolerance.";
            break;
        }
    }
    return OptimizationResult{x, f.evaluate(x), k, converged, msg};
}
