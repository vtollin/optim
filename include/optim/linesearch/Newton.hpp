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
class Newton : public LineSearchBase<optim::TwiceDifferentiableFunction> {
  public:
    explicit Newton(StepLengthMethod method = StepLengthMethod::ARMIJO, int max_iterations = 1000,
                    optim::ConvergenceCriteria criteria = {},
                    optim::logger::Logger *logger = nullptr);

    explicit Newton(std::unique_ptr<StepLengthPolicy> policy, int max_iterations = 1000,
                    optim::ConvergenceCriteria criteria = {},
                    optim::logger::Logger *logger = nullptr);

  private:
    Eigen::VectorXd computeDirection(const TwiceDifferentiableFunction &f, const Eigen::VectorXd &x,
                                     const Eigen::VectorXd &grad) override;
    CholeskyFactor modifiedCholesky(const Eigen::MatrixXd &hessian);
};
} // namespace optim::linesearch
