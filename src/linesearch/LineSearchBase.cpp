#include "optim/linesearch/LineSearchBase.hpp"
#include "optim/Functions.hpp"
#include "optim/linesearch/ArmijoBacktracking.hpp"
#include "optim/linesearch/StepLengthPolicy.hpp"
#include "optim/linesearch/StrongWolfe.hpp"
#include "optim/logger/Logger.hpp"
#include <stdexcept>

using namespace optim::linesearch;

template <typename FuncType>
LineSearchBase<FuncType>::LineSearchBase(std::unique_ptr<StepLengthPolicy> step_length_policy,
                                         int max_iterations, optim::ConvergenceCriteria criteria,
                                         optim::logger::Logger *logger)
    : OptimizerBase(max_iterations, criteria, logger)
    , step_length_policy_(std::move(step_length_policy)) {
    step_length_policy_->setLogger(logger);
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
    std::string msg = "Failed to converge in specified iterations.";
    int k = 0;
    resetState(f.sourceDimension());

    notifyStart();
    for (; k < max_iterations_; ++k) {
        if (grad.norm() < criteria_.grad_tol * (1.0 + x.norm())) { // relative tolerance
            converged = true;
            msg = "Converged: gradient norm fell below tolerance.";
            break;
        }

        Eigen::VectorXd direction = computeDirection(f, x, grad);
        StepResult res = step_length_policy_->computeStep(f, x, direction, grad);
        if (res.status == StepStatus::FAILURE) {
            msg = "Line search failed to find an acceptable step.";
            break;
        }

        Eigen::VectorXd step = res.alpha * direction;
        double f0 = f.evaluate(x);
        notifyIteration({k, f0, x, grad, step}); // rho/delta/accepted default to nullopt
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

        Eigen::VectorXd grad_new = f.gradient(x);
        updateState(step, grad_new - grad); // HOOK (BFGS only)
        grad = grad_new;
    }
    notifyFinish();
    return OptimizationResult{x, f.evaluate(x), k, converged, msg};
}

template <typename FuncType>
void LineSearchBase<FuncType>::setLogger(optim::logger::Logger *logger) {
    OptimizerBase::setLogger(logger);
    step_length_policy_->setLogger(logger);
}

template class LineSearchBase<optim::DifferentiableFunction>;
template class LineSearchBase<optim::TwiceDifferentiableFunction>;