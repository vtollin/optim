#pragma once
#include "optim/OptimizationResult.hpp"
#include "optim/linesearch/LineSearchBase.hpp"
#include "optim/linesearch/StepLengthMethod.hpp"
#include <Eigen/Dense>

namespace optim::logger {
class Logger;
}

namespace optim {
class TwiceDifferentiableFunction;
}

namespace optim::linesearch {
struct CholeskyFactor {
    Eigen::MatrixXd L;
    Eigen::VectorXd d;
};

struct CholeskyDiagnostics {
    Eigen::VectorXd d;   // modified diagonal of the LDL^T factorization
    double max_shift;    // max positive shift applied to any diagonal to enforce PD
};

class NewtonObserver {
  public:
    virtual ~NewtonObserver() = default;
    virtual void onModifiedCholesky(const CholeskyDiagnostics &) {}
};

class Newton : public LineSearchBase<optim::TwiceDifferentiableFunction> {
  public:
    explicit Newton(StepLengthMethod method = StepLengthMethod::ARMIJO, int max_iterations = 1000,
                    optim::ConvergenceCriteria criteria = {},
                    optim::logger::Logger *logger = nullptr);

    explicit Newton(std::unique_ptr<StepLengthPolicy> policy, int max_iterations = 1000,
                    optim::ConvergenceCriteria criteria = {},
                    optim::logger::Logger *logger = nullptr);

    // Does NOT take ownership; caller must keep the observer alive for the optimizer's lifetime.
    void setNewtonObserver(NewtonObserver *obs) { newton_obs_ = obs; }

  private:
    Eigen::VectorXd computeDirection(const TwiceDifferentiableFunction &f, const Eigen::VectorXd &x,
                                     const Eigen::VectorXd &grad) override;
    CholeskyFactor modifiedCholesky(const Eigen::MatrixXd &hessian);

    NewtonObserver *newton_obs_ = nullptr;
};
} // namespace optim::linesearch
