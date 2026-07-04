#include "optim/linesearch/SteepestDescent.hpp"
#include "optim/Functions.hpp"
#include "optim/linesearch/StepLengthMethod.hpp"
#include "optim/linesearch/StepLengthPolicy.hpp"
#include <Eigen/Dense>

using namespace optim::linesearch;
using optim::DifferentiableFunction;
using optim::OptimizationResult;

SteepestDescent::SteepestDescent(StepLengthMethod method, int max_iterations,
                                 optim::ConvergenceCriteria criteria)
    : SteepestDescent(makeStepLengthPolicy(method), max_iterations, criteria) {
}

SteepestDescent::SteepestDescent(std::unique_ptr<StepLengthPolicy> policy, int max_iterations,
                                 optim::ConvergenceCriteria criteria)
    : LineSearchBase(std::move(policy), max_iterations, criteria) {
}

// Search direction is simply negative gradient. N&W Section 3.1, p. 30.
Eigen::VectorXd SteepestDescent::computeDirection(const DifferentiableFunction &f,
                                                  const Eigen::VectorXd &x,
                                                  const Eigen::VectorXd &grad) {
    return -grad;
}
