#include "optim/trustregion/MoreSorensen.hpp"
#include "internal/BFGSHandler.hpp"
#include "internal/BMatrixHandler.hpp"
#include "internal/ExactHessianHandler.hpp"
#include "optim/Functions.hpp"
#include "optim/OptimizationResult.hpp"
#include "optim/OptimizerUtility.hpp"
#include "optim/logger/Logger.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>
using namespace optim::trustregion;
using optim::OptimizationResult;
using optim::TwiceDifferentiableFunction;

MoreSorensen::MoreSorensen(int max_iterations, optim::ConvergenceCriteria criteria,
                           double delta_init, double delta_max, double eta, BMatrixConfig cfg,
                           optim::logger::Logger *logger)
    : TrustRegionBase(max_iterations, criteria, delta_init, delta_max, eta, logger) {
    if (cfg == BMatrixConfig::EXACT) {
        b_handler_ = std::make_unique<ExactHessianHandler>();
    } else if (cfg == BMatrixConfig::APPROXIMATE) {
        b_handler_ = std::make_unique<BFGSHandler>();
    }
}

MoreSorensen::~MoreSorensen() = default;

OptimizationResult MoreSorensen::optimize(const TwiceDifferentiableFunction &f,
                                          const Eigen::VectorXd &x0) {
    int k = 0;
    bool converged = false;
    StopReason reason = StopReason::MAX_ITERS_REACHED;

    Eigen::VectorXd x = x0;

    int n = f.sourceDimension();
    if (n != x.size()) {
        throw std::invalid_argument(
            "[MoreSorensen] Initial vector is not in the source of objective function.");
    }
    if (delta_ < 0.0) {
        delta_ = std::min(1.0, f.gradient(x0).norm());
    }

    Eigen::VectorXd step(n);
    QuadraticModel m;
    m.B = b_handler_->initialize(f, x);
    for (; k < max_iterations_; ++k) {
        m.g = f.gradient(x);
        if (m.g.norm() < criteria_.grad_tol) {
            converged = true;
            reason = StopReason::GRADIENT_CONVERGED;
            break;
        }
        if (tryNewton(m.g, m.B, step)) {
            // step set to newton step
        } else {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(m.B);

            if (eigenSolver.info() != Eigen::Success) {
                throw std::runtime_error("[MoreSorensen] Eigen decomposition failed.");
            }

            Eigen::VectorXd eigenvalues = eigenSolver.eigenvalues();
            Eigen::MatrixXd eigenvectors = eigenSolver.eigenvectors();

            double lambda1 = eigenvalues(0);
            int multiplicity = 1;

            // Determine algebraic multiplictiy of smallest eigenvalue
            for (int i = 1; i < n; ++i) {
                if (std::abs(eigenvalues(i) - lambda1) < optim::utility::epsilon) {
                    ++multiplicity;
                } else {
                    break;
                }
            }

            bool hardCase = true;
            for (int i = 0; i < multiplicity; ++i) {
                double proj = eigenvectors.col(i).dot(m.g);
                if (std::abs(proj) > optim::utility::epsilon) {
                    hardCase = false;
                    break;
                }
            }

            double lambda;
            if (hardCase) {
                lambda = -lambda1;
                double sum = 0;
                Eigen::VectorXd ortho_vec = Eigen::VectorXd::Zero(n);
                for (int i = multiplicity; i < n; ++i) {
                    double num = eigenvectors.col(i).dot(m.g);
                    double deno = eigenvalues(i) + lambda;
                    double term = num / deno;
                    ortho_vec += term * eigenvectors.col(i);
                    sum += term * term;
                }
                double arg = delta_ * delta_ - sum;
                double tau = std::sqrt(std::max(arg, 0.0));
                Eigen::VectorXd z = eigenvectors.col(0);
                step = ortho_vec + tau * z;
            } else {
                step = newtonRootFind(m.B, m.g, lambda1);
            }
        }
        if (step.norm() < criteria_.step_tol * (x.norm() + 1.0)) {
            converged = true;
            reason = StopReason::STEP_STALLED;
            x += step;
            break;
        }
        if (criteria_.f_tol.has_value()) {
            double f0 = f.evaluate(x);
            double f1 = f.evaluate(x + step);
            if (std::abs(f1 - f0) < criteria_.f_tol.value() * (std::abs(f0) + 1.0)) {
                converged = true;
                reason = StopReason::F_CHANGE_BELOW_TOL;
                x += step;
                break;
            }
        }
        UpdateResult result = update(f, m, x, step);
        if (result.accepted) {
            x = x + step;
            m.B = b_handler_->getB(f, x);
        }
    }
    return OptimizationResult{x, f.evaluate(x), k, converged, reason};
}

bool MoreSorensen::tryNewton(const Eigen::VectorXd &grad, const Eigen::MatrixXd &B,
                             Eigen::VectorXd &p_out) {
    Eigen::LLT<Eigen::MatrixXd> llt(B);
    if (llt.info() == Eigen::Success) {
        Eigen::VectorXd p_newton = llt.solve(-grad);
        if (p_newton.norm() <= delta_) {
            p_out = p_newton;
            return true;
        }
    }
    return false;
}

Eigen::VectorXd MoreSorensen::newtonRootFind(const Eigen::MatrixXd &B, const Eigen::VectorXd &grad,
                                             double lambda1) {
    double lambda_precision = optim::utility::epsilon * (1.0 + std::abs(lambda1));
    double lambda = std::max(0.0, -lambda1 + lambda_precision);
    double tolerance = 1e-4 * delta_; // loose tolerance
    int max_iters = 3;                // log or expose?
    int n = B.col(0).size();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    Eigen::VectorXd p;
    int i = 0;
    while (i < max_iters) {
        Eigen::LLT<Eigen::MatrixXd> llt(B + lambda * I);
        while (llt.info() != Eigen::Success) { // bump lambda until success
            lambda = std::max(2 * lambda, lambda + 1.0);
            llt.compute(B + lambda * I);
        }

        p = llt.solve(-grad);
        double pnorm = p.norm();
        if (std::abs(pnorm - delta_) < tolerance) {
            return p;
        }

        Eigen::MatrixXd L = llt.matrixL();
        Eigen::VectorXd q = L.triangularView<Eigen::Lower>().solve(p);
        double qnorm = q.norm();

        lambda = lambda + (pnorm / qnorm) * (pnorm / qnorm) * ((pnorm - delta_) / delta_);
        lambda = std::max(lambda, -lambda1 + lambda_precision);
        ++i;
    }
    return (B + lambda * I).llt().solve(-grad);
}
