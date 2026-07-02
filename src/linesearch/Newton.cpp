#include "optim/linesearch/Newton.hpp"
#include "optim/Functions.hpp"
#include "optim/OptimizationResult.hpp"
#include "optim/OptimizerUtility.hpp"
#include "optim/linesearch/LineSearchBase.hpp"
#include "optim/linesearch/StepLengthMethod.hpp"
#include "optim/linesearch/StepLengthPolicy.hpp"
#include "optim/logger/Logger.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>

using namespace optim::linesearch;
using optim::OptimizationResult;
using optim::TwiceDifferentiableFunction;

Newton::Newton(StepLengthMethod method, int max_iterations, optim::ConvergenceCriteria criteria,
               optim::logger::Logger *logger)
    : Newton(makeStepLengthPolicy(method), max_iterations, criteria, logger) {
}

Newton::Newton(std::unique_ptr<StepLengthPolicy> policy, int max_iterations,
               optim::ConvergenceCriteria criteria, optim::logger::Logger *logger)
    : LineSearchBase(std::move(policy), max_iterations, criteria, logger) {
}

// Computes search direction by solving the Cholesky factorized system LDL^T y = -g in three steps.
Eigen::VectorXd Newton::computeDirection(const TwiceDifferentiableFunction &f,
                                         const Eigen::VectorXd &x, const Eigen::VectorXd &grad) {
    CholeskyFactor factor = modifiedCholesky(f.hessian(x));
    Eigen::VectorXd y = factor.L.triangularView<Eigen::Lower>().solve(-grad); // Ly = -g
    Eigen::VectorXd z = y.array() / factor.d.array();                         // Dz = y
    return factor.L.triangularView<Eigen::Lower>().transpose().solve(z);      // L^Tp = z
}

// Computes a modified LDL^T factorization where each diagonal d(j) is bumped up as needed to
// ensure positive definiteness. Guarantees a descent direction even when the Hessian is indefinite.
// Gill-Murray-Wright modified Cholesky. N&W Algorithm 3.4, p. 52.
CholeskyFactor Newton::modifiedCholesky(const Eigen::MatrixXd &hessian) {
    int n = hessian.rows();
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd d(n);
    double delta = 10e-3; // minimum ensures result is sufficiently positive definite
    double gamma = 0.0;
    double xi = 0.0;
    for (Eigen::Index j = 0; j < n; ++j) {
        gamma = std::max(gamma, std::abs(hessian(j, j)));
        for (Eigen::Index i = j + 1; i < n; ++i) {
            xi = std::max(xi, std::abs(hessian(i, j)));
        }
    }
    double beta_sq = std::max(std::max(gamma, xi / std::sqrt(n * n - 1)),
                              std::numeric_limits<double>::epsilon());
    for (int j = 0; j < n; ++j) { // column
        double theta = 0.0;
        Eigen::VectorXd Lj = L.row(j).segment(0, j);
        Eigen::VectorXd dvec = d.segment(0, j);
        double sum_outer = (dvec.array() * Lj.array().square()).sum();
        double cjj = hessian(j, j) - sum_outer;
        Eigen::VectorXd cij_vec = Eigen::VectorXd::Zero(n - j - 1);
        for (int i = j + 1; i < n; ++i) {
            Eigen::VectorXd Li = L.row(i).segment(0, j);
            double sum_inner = (dvec.array() * Li.array() * Lj.array()).sum();
            double cij = hessian(i, j) - sum_inner;
            if (std::abs(cij) > theta) {
                theta = std::abs(cij);
            }
            cij_vec(i - j - 1) = cij;
        }
        d(j) = std::max(std::max(std::abs(cjj), std::pow(theta, 2) / beta_sq), delta);
        for (int i = j + 1; i < n; ++i) {
            L(i, j) = cij_vec(i - j - 1) / d(j);
        }
        L(j, j) = 1.0;
    }
    return CholeskyFactor{L, d};
}
