#pragma once
#include "optim/OptimizationResult.hpp"
#include "optim/linesearch/LineSearchBase.hpp"
#include "optim/linesearch/StepLengthMethod.hpp"
#include <Eigen/Dense>

namespace optim::logger {
class Logger;
}

namespace optim {
class DifferentiableFunction;
} // namespace optim

namespace optim::linesearch {
struct BFGSUpdateInfo {
    bool was_damped; // true when Powell damping triggered (sy_k < 0.2 * sHs)
    double theta;    // damping factor (1.0 if not damped)
    double sy_k;     // original s^T y_k before damping
    double sHs;      // s^T H s
    double rho;      // 1 / s^T y after damping: curvature scale used in rank-2 update
};

class BFGSObserver {
  public:
    virtual ~BFGSObserver() = default;
    virtual void onBFGSUpdate(const BFGSUpdateInfo &) {}
};

class BFGS : public LineSearchBase<optim::DifferentiableFunction> {
  public:
    // Strong Wolfe is the default: the curvature condition it enforces implies s^T y > 0,
    // which is what makes the BFGS update maintain positive definiteness.
    // Armijo is permitted because Powell damping in updateBFGS provides a fallback.
    // However, convergence may be slower and the theoretical guarantees are weaker.
    explicit BFGS(StepLengthMethod method = StepLengthMethod::STRONG_WOLFE,
                  int max_iterations = 1000, optim::ConvergenceCriteria criteria = {},
                  optim::logger::Logger *logger = nullptr);

    explicit BFGS(std::unique_ptr<StepLengthPolicy> policy, int max_iterations = 1000,
                  optim::ConvergenceCriteria criteria = {},
                  optim::logger::Logger *logger = nullptr);

    // Does NOT take ownership; caller must keep the observer alive for the optimizer's lifetime.
    void setBFGSObserver(BFGSObserver *obs) { bfgs_obs_ = obs; }

  private:
    Eigen::VectorXd computeDirection(const DifferentiableFunction &f, const Eigen::VectorXd &x,
                                     const Eigen::VectorXd &grad) override;
    void resetState(int n) override;
    void updateState(const Eigen::VectorXd &s, const Eigen::VectorXd &y) override;
    void updateBFGS(const Eigen::VectorXd &s, const Eigen::VectorXd &y_k);
    Eigen::MatrixXd H_; // BFGS matrix state
    BFGSObserver *bfgs_obs_ = nullptr;
};
} // namespace optim::linesearch
