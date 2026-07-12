#include "optim/trustregion/TrustRegionBase.hpp"
#include "optim/Functions.hpp"
#include "optim/OptimizationResult.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <limits>

using namespace optim::trustregion;
using optim::TwiceDifferentiableFunction;

optim::OptimizationResult TrustRegionBase::optimize(const TwiceDifferentiableFunction &f,
                                                    const Eigen::VectorXd &x0) {
    if (x0.size() != f.sourceDimension()) {
        throw std::invalid_argument("[TrustRegion] x0 not in source of objective.");
    }

    Eigen::VectorXd x = x0;
    bool converged = false;
    StopReason reason = StopReason::MAX_ITERS_REACHED;
    int k = 0;
    Eigen::VectorXd grad = f.gradient(x);

    const double xscale = std::max(1.0, x0.norm());
    double delta_max;
    if (config_.delta_max.has_value()) {
        delta_max = config_.delta_max.value();
    } else if (config_.delta_init.has_value()) {
        delta_max = std::max(1e3 * xscale, 1e3 * *config_.delta_init);
    } else {
        delta_max = 1e3 * xscale;
    }
    double delta = config_.delta_init.value_or(xscale);
    delta = std::min(delta, delta_max);

    Eigen::MatrixXd B = initializeB(f, x);

    for (; k < max_iterations_; ++k) {
        if (grad.norm() < criteria_.grad_tol) {
            converged = true;
            reason = StopReason::GRADIENT_CONVERGED;
            break;
        }
        SubproblemResult res = solveSubproblem(grad, B, delta);
        Eigen::VectorXd step = res.p;
        double rho = computeRho(f, x, grad, B, step);
        if (rho < 0.25) {
            delta = 0.25 * delta;
        } else {
            if (rho > 0.75 && res.status == SubproblemStatus::BOUNDARY) {
                delta = std::min(2 * delta, delta_max);
            }
        }
        bool accepted;
        if (rho > config_.eta) {
            accepted = true;
            double f_x = f.evaluate(x);
            if (step.norm() < criteria_.step_tol * (1.0 + x.norm())) {
                converged = true;
                reason = StopReason::STEP_STALLED;
                x += step;
                break;
            }
            if (criteria_.f_tol.has_value()) {
                double f1 = f.evaluate(x + step);
                if (std::abs(f1 - f_x) < criteria_.f_tol.value() * (std::abs(f_x) + 1.0)) {
                    converged = true;
                    reason = StopReason::F_CHANGE_BELOW_TOL;
                    x += step;
                    break;
                }
            }
            x = x + step;
            grad = f.gradient(x); // only recomputed if step is accepted
            B = updateB(f, x);
        } else {
            accepted = false;
        }
    }
    return OptimizationResult{x, f.evaluate(x), k, converged, reason};
}

double TrustRegionBase::computeRho(const TwiceDifferentiableFunction &f, const Eigen::VectorXd &x,
                                   const Eigen::VectorXd &grad, const Eigen::MatrixXd &B,
                                   const Eigen::VectorXd &step) {
    double rho;
    double f_x = f.evaluate(x);
    double actual = f_x - f.evaluate(x + step);
    double predicted = -grad.dot(step) - 0.5 * step.dot(B * step);
    if (std::abs(predicted) > std::numeric_limits<double>::epsilon() * (1.0 + std::abs(f_x))) {
        rho = actual / predicted;
    } else { // model predicts almost no decrease -> near convergence (grad ~ 0)
        rho = 1.0;
    }
    return rho;
}

Eigen::MatrixXd TrustRegionBase::initializeB(const TwiceDifferentiableFunction &f,
                                             const Eigen::VectorXd &x0) {
    return f.hessian(x0);
}

Eigen::MatrixXd TrustRegionBase::updateB(const TwiceDifferentiableFunction &f,
                                         const Eigen::VectorXd &x) {
    return f.hessian(x);
}
