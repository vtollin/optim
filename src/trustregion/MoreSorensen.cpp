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

MoreSorensen::MoreSorensen(int max_iterations, optim::ConvergenceCriteria criteria,
                           double delta_init, double delta_max, double eta, BMatrixConfig cfg)
    : TrustRegionBase(max_iterations, criteria, delta_init, delta_max, eta) {
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
    int n = grad.size();
    Eigen::VectorXd step(n);
    SubproblemStatus status;

    if (tryNewton(grad, B, delta, step)) {
        status = SubproblemStatus::INTERIOR;
    } else {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(B);

        if (eigenSolver.info() != Eigen::Success) {
            throw std::runtime_error("[MoreSorensen] Eigen decomposition failed.");
        }

        Eigen::VectorXd eigenvalues = eigenSolver.eigenvalues();
        Eigen::MatrixXd eigenvectors = eigenSolver.eigenvectors();

        double lambda1 = eigenvalues(0);
        int multiplicity = 1;

        // Determine algebraic multiplictiy of smallest eigenvalue
        for (int i = 1; i < n; ++i) {
            if (std::abs(eigenvalues(i) - lambda1) < std::numeric_limits<double>::epsilon()) {
                ++multiplicity;
            } else {
                break;
            }
        }

        bool hardCase = true;
        for (int i = 0; i < multiplicity; ++i) {
            double proj = eigenvectors.col(i).dot(grad);
            if (std::abs(proj) > std::numeric_limits<double>::epsilon()) {
                hardCase = false;
                break;
            }
        }

        if (hardCase) {
            double lambda = -lambda1;
            double sum = 0;
            Eigen::VectorXd ortho_vec = Eigen::VectorXd::Zero(n);
            for (int i = multiplicity; i < n; ++i) {
                double num = eigenvectors.col(i).dot(grad);
                double deno = eigenvalues(i) + lambda;
                double term = num / deno;
                ortho_vec += term * eigenvectors.col(i);
                sum += term * term;
            }
            double arg = delta * delta - sum;
            double tau = std::sqrt(std::max(arg, 0.0));
            Eigen::VectorXd z = eigenvectors.col(0);
            step = ortho_vec + tau * z;
            status = SubproblemStatus::HARDCASE;
        } else {
            step = newtonRootFind(B, grad, lambda1, delta);
            status = SubproblemStatus::BOUNDARY;
        }
    }

    return SubproblemResult{step, status};
}

bool MoreSorensen::tryNewton(const Eigen::VectorXd &grad, const Eigen::MatrixXd &B, double delta,
                             Eigen::VectorXd &p_out) {
    Eigen::LLT<Eigen::MatrixXd> llt(B);
    if (llt.info() == Eigen::Success) {
        Eigen::VectorXd p_newton = llt.solve(-grad);
        if (p_newton.norm() <= delta) {
            p_out = p_newton;
            return true;
        }
    }
    return false;
}

Eigen::VectorXd MoreSorensen::newtonRootFind(const Eigen::MatrixXd &B, const Eigen::VectorXd &grad,
                                             double lambda1, double delta) {
    double lambda_precision = std::numeric_limits<double>::epsilon() * (1.0 + std::abs(lambda1));
    double lambda = std::max(0.0, -lambda1 + lambda_precision);
    double tolerance = 1e-4 * delta; // loose tolerance
    int max_iters = 3;               // log or expose?
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
        if (std::abs(pnorm - delta) < tolerance) {
            return p;
        }

        Eigen::MatrixXd L = llt.matrixL();
        Eigen::VectorXd q = L.triangularView<Eigen::Lower>().solve(p);
        double qnorm = q.norm();

        lambda = lambda + (pnorm / qnorm) * (pnorm / qnorm) * ((pnorm - delta) / delta);
        lambda = std::max(lambda, -lambda1 + lambda_precision);
        ++i;
    }
    return (B + lambda * I).llt().solve(-grad);
}
