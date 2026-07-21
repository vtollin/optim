#include "optim/trustregion/MoreSorensen.hpp"
#include "internal/BFGSHandler.hpp"
#include "internal/BMatrixHandler.hpp"
#include "internal/ExactHessianHandler.hpp"
#include "optim/Functions.hpp"
#include "optim/OptimizationResult.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
using namespace optim::trustregion;
using optim::TwiceDifferentiableFunction;

MoreSorensen::MoreSorensen(int max_iterations, BMatrixConfig cfg,
                           optim::ConvergenceCriteria criteria, TrustRegionConfig config)
    : TrustRegionBase(max_iterations, criteria, config) {
    if (cfg == BMatrixConfig::EXACT) {
        b_handler_ = std::make_unique<ExactHessianHandler>();
    } else if (cfg == BMatrixConfig::APPROXIMATE) {
        b_handler_ = std::make_unique<BFGSHandler>();
    }
}

MoreSorensen::~MoreSorensen() = default;

Eigen::MatrixXd MoreSorensen::initializeB(const TwiceDifferentiableFunction &f,
                                          const Eigen::VectorXd &x0) {
    return b_handler_->initialize(f, x0);
}

Eigen::MatrixXd MoreSorensen::updateB(const TwiceDifferentiableFunction &f,
                                      const Eigen::VectorXd &x) {
    return b_handler_->getB(f, x);
}

SubproblemResult MoreSorensen::solveSubproblem(const Eigen::VectorXd &grad,
                                               const Eigen::MatrixXd &B, double delta) {
    Eigen::Index n = grad.size();
    double hardCaseTol = 1e-8;
    // 1. Try lambda = 0 case
    Eigen::LLT<Eigen::MatrixXd> llt;
    llt.compute(B);
    if (llt.info() == Eigen::Success) {
        Eigen::VectorXd p = llt.solve(-grad);
        if (p.norm() <= delta) {
            bool onBoundary =
                (delta - p.norm()) < std::numeric_limits<double>::epsilon() * std::max(1.0, delta);
            return SubproblemResult{p, onBoundary ? SubproblemStatus::BOUNDARY
                                                  : SubproblemStatus::INTERIOR};
        }
    }

    double lambda_lo = std::max(0.0, -B.diagonal().minCoeff());
    double lambda_hi = grad.norm() / delta + B.norm(); // Frobenius norm
    double lambda = lambda_lo + std::numeric_limits<double>::epsilon() * std::max(lambda_lo, 1.0);
    double maxIters = 10;
    double primary_tol = 1e-3;
    for (int i = 0; i < maxIters; ++i) {
        Eigen::MatrixXd shifted = B;
        shifted.diagonal().array() += lambda;
        while (llt.compute(shifted).info() != Eigen::Success) {
            lambda_lo = std::max(lambda, lambda_lo);
            double lambda_new = std::max(std::sqrt(lambda_lo * lambda_hi),
                                         lambda_lo + 0.01 * (lambda_hi - lambda_lo));
            shifted.diagonal().array() += (lambda_new - lambda);
            lambda = lambda_new;
        }

        Eigen::VectorXd p = llt.solve(-grad);
        if (std::abs(p.norm() - delta) / delta <= primary_tol) {
            return SubproblemResult{p, SubproblemStatus::BOUNDARY};
        }
        if (p.norm() < delta) {
            lambda_hi = lambda;
        } else {
            lambda_lo = lambda;
        }
        // Newton root-find
        Eigen::VectorXd q = llt.matrixL().solve(p);
        lambda = lambda + std::pow((p.norm() / q.norm()), 2) * (p.norm() - delta) / delta;
        if (lambda <= lambda_lo || lambda >= lambda_hi) {
            lambda = std::max(lambda_lo + 0.01 * (lambda_hi - lambda_lo),
                              std::sqrt(lambda_lo * lambda_hi));
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(B);
    const Eigen::VectorXd &eigenvals = es.eigenvalues();
    const Eigen::MatrixXd &eigenvecs = es.eigenvectors();

    const double lambda1 = eigenvals(0);
    const Eigen::VectorXd qT_g = eigenvecs.transpose() * grad;
    Eigen::Index k = 1;
    double eigTol = 1e-10;
    while (k < n && std::abs(lambda1 - eigenvals(k)) < eigTol * std::max(1.0, std::abs(lambda1))) {
        k++;
    }
    bool g1_zero = true;
    for (Eigen::Index i = 0; i < k; ++i) {
        if (std::abs(qT_g(i)) > hardCaseTol * grad.norm()) {
            g1_zero = false;
            break;
        }
    }

    // construct squared norm of p(lambda1)
    double normSq_edge = 0.0;
    for (Eigen::Index i = k; i < n; ++i) {
        double d = eigenvals(i) - lambda1;
        normSq_edge += (qT_g(i) * qT_g(i)) / (d * d);
    }

    // If g is not orthogonal to q1 or the norm of p(lambda1) is greater than delta the solution
    // exists on (-lambda1, inf) (not hard case).
    if (g1_zero && normSq_edge <= delta * delta) {
        Eigen::VectorXd p_particular = Eigen::VectorXd::Zero(n);
        for (Eigen::Index j = k; j < n; ++j) {
            p_particular += (-qT_g(j) / (eigenvals(j) - lambda1)) * eigenvecs.col(j);
        }
        double tau = std::sqrt(std::max(delta * delta - p_particular.squaredNorm(), 0.0));
        Eigen::VectorXd p = p_particular + tau * eigenvecs.col(0);
        return SubproblemResult{p, SubproblemStatus::HARDCASE};
    } else {
        double fallback_tol = 1e-6;
        double lo = -lambda1;
        double hi = -lambda1 + grad.norm() / delta;
        lambda = std::max(lo + 0.01 * (hi - lo), std::sqrt(lo * hi));
        for (int i = 0; i < 10; ++i) {
            double p_norm2 = 0.0;
            double q_norm2 = 0.0;
            for (Eigen::Index j = 0; j < n; ++j) {
                double d = eigenvals(j) + lambda;
                double t = qT_g(j) / d;
                p_norm2 += t * t;
                q_norm2 += t * t / d;
            }
            double p_norm = std::sqrt(p_norm2);

            if (std::abs(p_norm - delta) / delta < fallback_tol) {
                break;
            }
            if (p_norm > delta) {
                lo = lambda;
            } else {
                hi = lambda;
            }

            lambda += (p_norm2 / q_norm2) * (p_norm - delta) / delta;
            if (lambda <= lo || lambda >= hi) {
                lambda = std::max(lo + 0.01 * (hi - lo), std::sqrt(lo * hi));
            }
        }
        Eigen::VectorXd p = -eigenvecs * (qT_g.array() / (eigenvals.array() + lambda)).matrix();
        return SubproblemResult{p, SubproblemStatus::BOUNDARY};
    }
}