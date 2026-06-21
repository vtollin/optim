#include "optim/linesearch/BFGS.hpp"
#include "optim/linesearch/LineSearchBase.hpp"
#include "optim/linesearch/StepLengthPolicy.hpp"
#include "optim/logger/Logger.hpp"
#include "optim/Functions.hpp"
#include "optim/OptimizationResult.hpp"
#include <Eigen/Dense>
#include <stdexcept>
#include <string>

using namespace optim::linesearch;
using optim::DifferentiableFunction;
using optim::OptimizationResult;

BFGS::BFGS(SearchStrategy search_strategy, int max_iterations, optim::ConvergenceCriteria criteria,
           std::shared_ptr<optim::logger::Logger> logger)
    : LineSearchBase(search_strategy, max_iterations, criteria, logger) {
    if (max_iterations_ < 1) {
        throw std::invalid_argument("[BFGS] max_iterations must be positive.");
    }
    if (criteria_.grad_tol <= 0 || criteria_.f_tol <= 0 || criteria_.step_tol <= 0) {
        throw std::invalid_argument("[BFGS] Tolerances must be positive.");
    }
}

// Maintains a quasi-Newton approximation to the Hessian updated each iteration via the BFGS
// rank-2 formula. Powell damping in updateBFGS enforces the curvature condition when the line
// search does not (e.g. Armijo). Strong Wolfe is the recommended default. N&W Section 6.1, p. 136.
OptimizationResult BFGS::optimize(const DifferentiableFunction &f, const Eigen::VectorXd &x0) {
    int n = f.sourceDimension();
    if (x0.size() != n) {
        throw std::invalid_argument(
            "[BFGS] Initial vector is not in the source of objective function.");
    }

    Eigen::VectorXd x = x0;
    bool converged = false;
    std::string msg = "Failed to converge in specified iterations.";
    int k = 0;
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(n, n);
    Eigen::VectorXd grad = f.gradient(x);
    for (; k < max_iterations_; ++k) {
        if (grad.norm() < criteria_.grad_tol * (1.0 + x.norm())) { // relative tolerance with absolute floor
            converged = true;
            msg = "Converged: gradient norm fell below tolerance.";
            break;
        }
        Eigen::VectorXd direction = -H * grad;
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
        Eigen::VectorXd next_grad = f.gradient(x);
        updateBFGS(H, step, next_grad - grad);
        grad = next_grad;
    }
    return OptimizationResult{x, f.evaluate(x), k, converged, msg};
}

// Applies the BFGS rank-2 update to the inverse Hessian approximation H_k. Powell damping
// blends y_k toward H_k s when the curvature condition is weak, keeping H_k positive definite.
// Adapted from N&W Algorithm 18.2, p. 537.
void BFGS::updateBFGS(Eigen::MatrixXd &H, const Eigen::VectorXd &s, const Eigen::VectorXd &y_k) {
    Eigen::VectorXd Hs = H * s;
    double sHs  = s.dot(Hs);
    double sy_k = s.dot(y_k);
    Eigen::VectorXd y;
    if (sy_k < 0.2 * sHs) { // Powell damping: enforce s^T y >= 0.2 * s^T H s
        double theta = 0.8 * sHs / (sHs - sy_k);
        y = theta * y_k + (1.0 - theta) * Hs;
    } else {
        y = y_k;
    }

    // H_{k+1} = (I - rho s y^T) H_k (I - rho y s^T) + rho s s^T
    double rho       = 1.0 / s.dot(y);
    Eigen::VectorXd Hy = H * y;
    double c         = 1.0 + rho * y.dot(Hy);
    H.noalias()     -= rho * Hy * s.transpose();
    H.noalias()     -= rho * s  * Hy.transpose();
    H.noalias()     += rho * c  * s * s.transpose();
}
