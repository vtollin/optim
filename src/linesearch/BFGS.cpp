#include "optim/linesearch/BFGS.hpp"
#include "optim/Functions.hpp"
#include "optim/OptimizationResult.hpp"
#include "optim/linesearch/LineSearchBase.hpp"
#include "optim/linesearch/StepLengthMethod.hpp"
#include "optim/linesearch/StepLengthPolicy.hpp"
#include "optim/logger/Logger.hpp"
#include <Eigen/Dense>
#include <stdexcept>
#include <string>

using namespace optim::linesearch;
using optim::DifferentiableFunction;
using optim::OptimizationResult;

BFGS::BFGS(StepLengthMethod method, int max_iterations, optim::ConvergenceCriteria criteria,
           optim::logger::Logger *logger)
    : BFGS(makeStepLengthPolicy(method), max_iterations, criteria, logger) {
}

BFGS::BFGS(std::unique_ptr<StepLengthPolicy> policy, int max_iterations,
           optim::ConvergenceCriteria criteria, optim::logger::Logger *logger)
    : LineSearchBase(std::move(policy), max_iterations, criteria, logger) {
    if (max_iterations_ < 1) {
        throw std::invalid_argument("[BFGS] max_iterations must be positive.");
    }
    if (criteria_.grad_tol <= 0 || criteria_.f_tol <= 0 || criteria_.step_tol <= 0) {
        throw std::invalid_argument("[BFGS] Tolerances must be positive.");
    }
}

Eigen::VectorXd BFGS::computeDirection(const DifferentiableFunction &f, const Eigen::VectorXd &x,
                                       const Eigen::VectorXd &grad) {
    return -H_ * grad;
}

void BFGS::resetState(int n) {
    H_ = Eigen::MatrixXd::Identity(n, n);
}

void BFGS::updateState(const Eigen::VectorXd &s, const Eigen::VectorXd &y) {
    updateBFGS(s, y);
}

// Applies the BFGS rank-2 update to the inverse Hessian approximation H_k. Powell damping
// blends y_k toward H_k s when the curvature condition is weak, keeping H_k positive definite.
// Adapted from N&W Algorithm 18.2, p. 537.
void BFGS::updateBFGS(const Eigen::VectorXd &s, const Eigen::VectorXd &y_k) {
    Eigen::VectorXd Hs = H_ * s;
    double sHs = s.dot(Hs);
    double sy_k = s.dot(y_k);
    Eigen::VectorXd y;
    if (sy_k < 0.2 * sHs) { // Powell damping: enforce s^T y >= 0.2 * s^T H s
        double theta = 0.8 * sHs / (sHs - sy_k);
        y = theta * y_k + (1.0 - theta) * Hs;
    } else {
        y = y_k;
    }

    // H_{k+1} = (I - rho s y^T) H_k (I - rho y s^T) + rho s s^T
    double rho = 1.0 / s.dot(y);
    Eigen::VectorXd Hy = H_ * y;
    double c = 1.0 + rho * y.dot(Hy);
    H_.noalias() -= rho * Hy * s.transpose();
    H_.noalias() -= rho * s * Hy.transpose();
    H_.noalias() += rho * c * s * s.transpose();
}
