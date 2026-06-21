#pragma once
#include "optim/linesearch/LineSearchBase.hpp"
#include "optim/linesearch/StepLengthMethod.hpp"
#include "optim/OptimizationResult.hpp"
#include <Eigen/Dense>

namespace optim::logger {
class Logger;
}

namespace optim { class TwiceDifferentiableFunction; }

namespace optim::linesearch {
struct CholeskyFactor {
    Eigen::MatrixXd L;
    Eigen::VectorXd d;
};
class Newton : public LineSearchBase<optim::TwiceDifferentiableFunction> {
  public:
    explicit Newton(StepLengthMethod method = StepLengthMethod::ARMIJO, int max_iterations = 1000,
                    optim::ConvergenceCriteria criteria = {},
                    std::shared_ptr<optim::logger::Logger> logger = nullptr);

    explicit Newton(std::unique_ptr<StepLengthPolicy> policy, int max_iterations = 1000,
                    optim::ConvergenceCriteria criteria = {},
                    std::shared_ptr<optim::logger::Logger> logger = nullptr);

    optim::OptimizationResult optimize(const optim::TwiceDifferentiableFunction &f,
                                       const Eigen::VectorXd &x0) override;

  private:
    CholeskyFactor modifiedCholesky(const Eigen::MatrixXd &hessian);
};
} // namespace optim::linesearch
