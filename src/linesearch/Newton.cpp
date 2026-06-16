#include "optim/linesearch/Newton.hpp"
#include "optim/linesearch/LineSearchBase.hpp"
#include "optim/linesearch/SearchStrategyBase.hpp"
#include "optim/logger/Logger.hpp"
#include "optim/AbstractFunctions.hpp"
#include "optim/OptimizationResult.hpp"
#include "optim/OptimizerUtility.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>

using namespace optim::linesearch;
using optim::abstract::TwiceDifferentiableFunction;
using optim::abstract::DifferentiableFunction;
using optim::OptimizationResult;

Newton::Newton(SearchStrategy search_strategy, int max_iterations,
               optim::ConvergenceCriteria criteria, std::shared_ptr<optim::logger::Logger> logger)
    : LineSearchBase(search_strategy, max_iterations, criteria, logger) {
    if (max_iterations_ < 1) {
        throw std::invalid_argument("[Newton] Max iterations must be positive.");
    }
    if (criteria_.grad_tol <= 0 || criteria_.f_tol <= 0 || criteria_.step_tol <= 0) {
        throw std::invalid_argument("[Newton] Tolerances must be positive.");
    }
}

// Computes search directions by solving the Newton system H*p = -g, regularizing the Hessian
// via modified Cholesky to guarantee a descent direction. Delegates step length to the
// configured line search strategy. N&W Section 3.4, pp. 48-49.
OptimizationResult Newton::optimize(const TwiceDifferentiableFunction &f,
                                    const Eigen::VectorXd &x0) {
    if (x0.size() != f.sourceDimension()) {
        throw std::invalid_argument(
            "[Newton] Initial vector is not in the source of objective function.");
    }

    Eigen::VectorXd x = x0;
    bool converged = false;
    std::string msg = "Failed to converge in specified iterations.";
    int k = 0;
    for (; k < max_iterations_; ++k) {
        Eigen::VectorXd grad = f.gradient(x);
        if (grad.norm() <
            criteria_.grad_tol * (1.0 + x.norm())) { // relative tolerance with absolute floor
            converged = true;
            msg = "Converged: gradient norm fell below tolerance.";
            break;
        }
        Eigen::MatrixXd hess = f.hessian(x);
        CholeskyFactor factor = modifiedCholesky(hess);
        Eigen::VectorXd y = factor.L.triangularView<Eigen::Lower>().solve(-grad); // Ly = -g
        Eigen::VectorXd z = y.array() / factor.d.array();                         // Dz = y
        Eigen::VectorXd direction =
            factor.L.triangularView<Eigen::Lower>().transpose().solve(z); // L^Tp = z

        double alpha = search_strategy_->computeStep(f, x, direction, grad);
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
    }
    return OptimizationResult{x, f.evaluate(x), k, converged, msg};
}

OptimizationResult Newton::optimize(const DifferentiableFunction &f, const Eigen::VectorXd &x0) {
    try {
        return optimize(dynamic_cast<const TwiceDifferentiableFunction &>(f), x0);
    } catch (const std::bad_cast &) {
        throw std::invalid_argument("[Newton] optimize() requires a TwiceDifferentiableFunction.");
    }
}

// Computes a modified LDL^T factorization where each diagonal d(j) is bumped up as needed to
// ensure positive definiteness. Guarantees a descent direction even when the Hessian is indefinite.
// Gill-Murray-Wright modified Cholesky. N&W Algorithm 3.4, p. 52.
CholeskyFactor Newton::modifiedCholesky(const Eigen::MatrixXd &hessian) {
    int n = hessian.rows();
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd d(n);
    double beta = 10.0;           // controls sensitivity to off-diagonal magnitude
    double delta = 10e-3;         // minimum ensures result is sufficiently positive definite
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
        d(j) = std::max(std::max(std::abs(cjj), std::pow(theta / beta, 2)), delta);
        for (int i = j + 1; i < n; ++i) {
            L(i, j) = cij_vec(i - j - 1) / d(j);
        }
        L(j, j) = 1.0;
    }
    return CholeskyFactor{L, d};
}
