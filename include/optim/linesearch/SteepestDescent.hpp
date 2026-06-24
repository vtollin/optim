#pragma once
#include "Eigen/Dense"
#include "optim/OptimizationResult.hpp"
#include "optim/linesearch/LineSearchBase.hpp"
#include "optim/linesearch/StepLengthMethod.hpp"

namespace optim::logger {
class Logger;
}

namespace optim {
class DifferentiableFunction;
}

namespace optim::linesearch {
class SteepestDescent : public LineSearchBase<optim::DifferentiableFunction> {
  public:
    explicit SteepestDescent(StepLengthMethod method = StepLengthMethod::ARMIJO,
                             int max_iterations = 1000, optim::ConvergenceCriteria criteria = {},
                             optim::logger::Logger *logger = nullptr);

    explicit SteepestDescent(std::unique_ptr<StepLengthPolicy> policy, int max_iterations = 1000,
                             optim::ConvergenceCriteria criteria = {},
                             optim::logger::Logger *logger = nullptr);

  private:
    Eigen::VectorXd computeDirection(const DifferentiableFunction &f, const Eigen::VectorXd &x,
                                     const Eigen::VectorXd &grad) override;
};
} // namespace optim::linesearch
