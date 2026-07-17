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

// TODO: Decide how to handle the hard case. The current bracket-collapse check is dead code since
// the outer loop is bound to 5 iterations. An alternative is add the eigendecomposition outside of
// the loop as a fallback. The O(n^3) cost will amortize, since the Cholesky-only path will converge
// in most cases. That is not true More-Sorensen though, which takes an entirely different
// approach. Worth considering other options.
SubproblemResult MoreSorensen::solveSubproblem(const Eigen::VectorXd &grad,
                                               const Eigen::MatrixXd &B, double delta) {
    Eigen::VectorXd p;
    Eigen::Index n = grad.size();
    // 1. Try lambda = 0 case
    Eigen::LLT<Eigen::MatrixXd> llt;
    llt.compute(B);
    if (llt.info() == Eigen::Success) {
        p = llt.solve(-grad);
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
    for (int i = 0; i < 5; ++i) { // fixed max iterations
        while (llt.compute(B + lambda * Eigen::MatrixXd::Identity(n, n)).info() !=
               Eigen::Success) { // might need termination
            lambda_lo = std::max(lambda, lambda_lo);
            lambda = std::max(std::sqrt(lambda_lo * lambda_hi),
                              lambda_lo + 0.01 * (lambda_hi - lambda_lo));
        }

        if (lambda_hi - lambda_lo <=
            std::numeric_limits<double>::epsilon() * std::max(1.0, lambda_hi)) {
            // hard case using full eigendecomposition. Does not handle geometric multiplicity > 1
            // of lambda1
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(B);
            const Eigen::VectorXd &eigenvals = es.eigenvalues();
            const Eigen::MatrixXd &eigenvecs = es.eigenvectors();

            double lambda1 = eigenvals(0);

            // build particular solution
            Eigen::VectorXd g_proj = eigenvecs.transpose() * grad;
            Eigen::VectorXd p_particular = Eigen::VectorXd::Zero(n);
            for (Eigen::Index j = 1; j < n; ++j) {
                p_particular += (-g_proj(j) / (eigenvals(j) - lambda1)) * eigenvecs.col(j);
            }
            double tau = std::sqrt(std::max(0.0, delta * delta - p_particular.squaredNorm()));
            Eigen::VectorXd z1 = eigenvecs.col(0);

            p = p_particular + tau * z1;
            return SubproblemResult{p, SubproblemStatus::HARDCASE};
        }

        p = llt.solve(-grad);
        if (std::abs(p.norm() - delta) / delta <= 1e-2) { // converged
            break;
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
    return SubproblemResult{p, SubproblemStatus::BOUNDARY};
}