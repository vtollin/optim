#include "optimization/Optimizer/Newton.hpp"
#include "optimization/LineSearch/SearchStrategyBase.hpp"
#include "optimization/Logger.hpp"
#include "optimization/ObjectiveFunctionBase.hpp"
#include "optimization/OptimizationResult.hpp"
#include "optimization/OptimizationUtils.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <string>

using namespace Optimizer;

Newton::Newton(std::shared_ptr<LineSearch::SearchStrategyBase> search_strategy, int max_iterations,
               double tol, std::shared_ptr<Logger> logger)
    : search_strategy_(search_strategy)
    , max_iterations_(max_iterations)
    , tol_(tol)
    , logger_(logger) {
    if (max_iterations_ < 1) {
        throw std::invalid_argument("[Newton]: Max iterations must be positive.");
    }
    if (tol_ <= 0) {
        throw std::invalid_argument("[Newton]: Tolerance must be positive.");
    }
}

OptimizationResult Newton::optimize(const ObjectiveFunctionBase &f, const Eigen::VectorXd &x0) {
    int k = 0;
    bool converged = false;
    std::string msg = "Failed to converge";
    Eigen::VectorXd x = x0;

    if (f.sourceDimension() != x.size()) {
        throw std::invalid_argument(
            "[Newton]: Initial vector is not in the source of objective function.");
    }

    for (; k < max_iterations_; ++k) {
        Eigen::VectorXd grad = f.gradient(x);
        if (grad.norm() < tol_) {
            converged = true;
            msg = "Converged successfully";
            break;
        }
        Eigen::MatrixXd hess = f.hessian(x);
        CholeskyFactor factor = modifiedCholesky(hess);
        Eigen::VectorXd direction = solveLDLT(factor.L, factor.d, grad);

        Eigen::VectorXd step = search_strategy_->computeStep(f, x, direction, grad);
        if (step.norm() == 0.0) {
            msg = "Search strategy returned 0 step.";
            break;
        }
        if (logger_) {
            logger_->logIteration(k, x, grad, f.evaluate(x), step);
        }
        x += step;
    }
    return OptimizationResult{x, f.evaluate(x), k, converged, msg};
}

CholeskyFactor Newton::modifiedCholesky(const Eigen::MatrixXd &hessian) {
    int n = hessian.rows();
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd d(n);
    double beta = 10.0;
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
        d(j) = std::max(std::max(std::abs(cjj), std::pow(theta / beta, 2)), Utility::sqrt_epsilon);
        for (int i = j + 1; i < n; ++i) {
            L(i, j) = cij_vec(i - j - 1) / d(j);
        }
        L(j, j) = 1.0;
    }
    return CholeskyFactor{L, d};
}

Eigen::VectorXd Newton::solveLDLT(const Eigen::MatrixXd &L, const Eigen::VectorXd &d,
                                  const Eigen::VectorXd &grad) {
    int n = grad.size();
    Eigen::VectorXd y(n); // solving Ly = -grad, lower triangular solve
    for (int i = 0; i < n; ++i) {
        if (i > 0) {
            y(i) = -grad(i) - L.row(i).segment(0, i).dot(y.segment(0, i));
        } else {
            y(i) = -grad(i);
        }
    }

    Eigen::VectorXd z = y.array() / d.array(); // solving Dz = y, diagonal solve

    Eigen::VectorXd p(n); // solving Ltp = z
    for (int i = n - 1; i >= 0; --i) {
        int len = n - i - 1;
        if (len > 0) {
            p(i) = z(i) - L.col(i).segment(i + 1, len).dot(p.segment(i + 1, len));
        } else {
            p(i) = z(i);
        }
    }

    return p;
}
