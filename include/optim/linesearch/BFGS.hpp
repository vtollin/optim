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

    optim::OptimizationResult optimize(const optim::DifferentiableFunction &f,
                                       const Eigen::VectorXd &x0) override;

  private:
    void updateBFGS(Eigen::MatrixXd &H, const Eigen::VectorXd &s, const Eigen::VectorXd &y_k);
};
} // namespace optim::linesearch
