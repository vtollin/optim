#include "optim/linesearch/SteepestDescent.hpp"
#include "optim/Functions.hpp"
#include "optim/OptimizerUtility.hpp"
#include "optim/linesearch/StepLengthMethod.hpp"
#include "optim/linesearch/StepLengthPolicy.hpp"
#include "optim/logger/Logger.hpp"
#include <Eigen/Dense>
#include <string>

using namespace optim::linesearch;
using optim::DifferentiableFunction;
using optim::OptimizationResult;

SteepestDescent::SteepestDescent(StepLengthMethod method, int max_iterations,
                                 optim::ConvergenceCriteria criteria, optim::logger::Logger *logger)
    : SteepestDescent(makeStepLengthPolicy(method), max_iterations, criteria, logger) {
}

SteepestDescent::SteepestDescent(std::unique_ptr<StepLengthPolicy> policy, int max_iterations,
                                 optim::ConvergenceCriteria criteria, optim::logger::Logger *logger)
    : LineSearchBase(std::move(policy), max_iterations, criteria, logger) {
    if (max_iterations_ < 1) {
        throw std::invalid_argument("[SteepestDescent] Max iterations must be positive.");
    }
    if (criteria_.grad_tol <= 0 || criteria_.f_tol <= 0 || criteria_.step_tol <= 0) {
        throw std::invalid_argument("[SteepestDescent] Tolerances must be positive.");
    }
}

// Search direction is simply negative gradient. N&W Section 3.1, p. 30.
Eigen::VectorXd SteepestDescent::computeDirection(const DifferentiableFunction &f,
                                                  const Eigen::VectorXd &x,
                                                  const Eigen::VectorXd &grad) {
    return -grad;
}
