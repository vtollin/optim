#include "optimization/Optimizer/BFGS.hpp"
#include "optimization/LineSearch/SearchStrategyBase.hpp"
#include "optimization/ObjectiveFunctionBase.hpp"
#include "optimization/OptimizationResult.hpp"
#include <Eigen/Dense>
#include <stdexcept>
#include <string>

using namespace Optimizer;

BFGS::BFGS(std::shared_ptr<LineSearch::SearchStrategyBase> search_strategy, int max_iterations,
           double tol, std::shared_ptr<Logger> logger)
    : search_strategy_(search_strategy)
    , max_iterations_(max_iterations)
    , tol_(tol)
    , logger_(logger) {
    if (max_iterations_ < 1) {
        throw std::invalid_argument("[BFGS]: max_iterations must be > 1");
    }
    if (tol_ <= 0) {
        throw std::invalid_argument("[BFGS]: tolerance must be positive");
    }
}

OptimizationResult BFGS::optimize(const ObjectiveFunctionBase &f, const Eigen::VectorXd &x0) {
    int k = 0;
    bool converged = false;
    std::string msg = "Failed to converge";
    Eigen::VectorXd x = x0;
    int n = f.sourceDimension();
    if (x.size() != n) {
        throw std::invalid_argument(
            "[BFGS]: Initial vector is not in the source of objective function.");
    }
    Eigen::MatrixXd B = Eigen::MatrixXd::Identity(n, n);
    Eigen::VectorXd grad = f.gradient(x);
    for (; k < max_iterations_; ++k) {
        if (grad.norm() < tol_) {
            converged = true;
            msg = "Converged successfully";
            break;
        }
        Eigen::VectorXd direction = -B * grad;
        Eigen::VectorXd step = search_strategy_->computeStep(f, x, direction, grad);
        if (step.norm() == 0.0) {
            msg = "Search strategy returned 0 step.";
            break;
        }
        Eigen::VectorXd next_x = x + step;
        Eigen::VectorXd next_grad = f.gradient(next_x);
        // scaled I when initially or when reset
        updateBFGS(B, step, next_grad - grad);
        x = next_x;
        grad = next_grad;
    }
    return OptimizationResult{x, f.evaluate(x), k, converged, msg};
}

void BFGS::updateBFGS(Eigen::MatrixXd &B_k, const Eigen::VectorXd &s_k,
                      const Eigen::VectorXd &y_k) {
    double sT_y = s_k.dot(y_k);
    Eigen::VectorXd Bs = B_k * s_k;
    double sBs = (Bs).dot(s_k);
    double delta = 0.2;
    Eigen::VectorXd y;
    if (sT_y < delta * sBs) { // Powell's correction
        double theta = (1 - delta) * sBs / (sBs - sT_y);
        y = theta * y_k + (1 - theta) * Bs;
    } else {
        y = y_k;
    }

    double gamma = 1.0 / s_k.dot(y);
    Eigen::VectorXd u = B_k * y;
    // split into 3 rank-1 updates
    B_k.noalias() -= gamma * u * s_k.transpose();
    B_k.noalias() -= gamma * s_k * u.transpose();

    double c = 1.0 + gamma * y.dot(u);
    B_k.noalias() += gamma * c * (s_k * s_k.transpose());
}
