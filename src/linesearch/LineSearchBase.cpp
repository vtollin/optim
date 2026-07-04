#include "optim/linesearch/LineSearchBase.hpp"
#include "optim/Functions.hpp"
#include "optim/linesearch/ArmijoBacktracking.hpp"
#include "optim/linesearch/StepLengthPolicy.hpp"
#include "optim/linesearch/StrongWolfe.hpp"
#include <stdexcept>

using namespace optim::linesearch;

template <typename FuncType>
LineSearchBase<FuncType>::LineSearchBase(std::unique_ptr<StepLengthPolicy> step_length_policy,
                                         int max_iterations, optim::ConvergenceCriteria criteria)
    : OptimizerBase(max_iterations, criteria)
    , step_length_policy_(std::move(step_length_policy)) {
    if (max_iterations_ < 1) {
        throw std::invalid_argument("[LineSearch] max_iterations must be positive.");
    }
    double f_tol = 1;
    if (criteria_.f_tol.has_value()) {
        f_tol = criteria_.f_tol.value();
    }
    if (criteria_.grad_tol <= 0 || f_tol <= 0 || criteria_.step_tol <= 0) {
        throw std::invalid_argument("[LineSearch] Tolerances must be positive.");
    }
}

template <typename FuncType>
optim::OptimizationResult LineSearchBase<FuncType>::optimize(const FuncType &f,
                                                             const Eigen::VectorXd &x0) {
    if (x0.size() != f.sourceDimension()) {
        throw std::invalid_argument("[LineSearch] x0 not in source of objective.");
    }

    Eigen::VectorXd x = x0;
    Eigen::VectorXd grad = f.gradient(x); // carried as loop state
    bool converged = false;
    StopReason reason = StopReason::MAX_ITERS_REACHED;
    int k = 0;
    resetState(f.sourceDimension());

    notifyStart();
    for (; k < max_iterations_; ++k) {
        if (grad.norm() < criteria_.grad_tol * (1.0 + x.norm())) { // relative tolerance
            converged = true;
            reason = StopReason::GRADIENT_CONVERGED;
            break;
        }

        Eigen::VectorXd direction = computeDirection(f, x, grad);
        StepResult res = step_length_policy_->computeStep(f, x, direction, grad);
        if (res.status == StepStatus::FAILURE) {
            reason = StopReason::LINE_SEARCH_FAILURE;
            break;
        }

        Eigen::VectorXd step = res.alpha * direction;
        double f0 = f.evaluate(x);
        notifyIteration({k, f0, x, grad, step, res.status});
        x += step;
        if (step.norm() < criteria_.step_tol * (1.0 + x.norm())) {
            reason = StopReason::STEP_STALLED;
            break;
        }
        if (criteria_.f_tol.has_value()) {
            double f1 = f.evaluate(x);
            if (std::abs(f1 - f0) < criteria_.f_tol.value() * (std::abs(f0) + 1.0)) {
                reason = StopReason::F_CHANGE_BELOW_TOL;
                break;
            }
        }

        Eigen::VectorXd grad_new = f.gradient(x);
        updateState(step, grad_new - grad); // HOOK (BFGS only)
        grad = grad_new;
    }
    notifyFinish();
    return OptimizationResult{x, f.evaluate(x), k, converged, reason};
}

namespace optim::linesearch {
template class LineSearchBase<optim::DifferentiableFunction>;
template class LineSearchBase<optim::TwiceDifferentiableFunction>;
} // namespace optim::linesearch