#include "optim/OptimizerBase.hpp"

using namespace optim;

OptimizerBase::OptimizerBase(int max_iterations, ConvergenceCriteria criteria)
    : max_iterations_(max_iterations), criteria_(criteria) {
    if (max_iterations_ < 1) {
        throw std::invalid_argument("[Optimizer] max_iterations must be at least 1.");
    }
    if (criteria_.f_tol.has_value()) {
        if (criteria_.f_tol.value() <= 0) {
            throw std::invalid_argument("[Optimizer] f_tol must be positive");
        }
    }
    if (criteria_.step_tol.has_value()) {
        if (criteria_.step_tol.value() <= 0) {
            throw std::invalid_argument("[Optimizer] step_tol must be positive.");
        }
    }
    if (criteria_.grad_tol <= 0) {
        throw std::invalid_argument("[Optimizer] grad_tol must be positive.");
    }
}
