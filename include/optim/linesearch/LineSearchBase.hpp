#pragma once
#include "optim/OptimizationResult.hpp"
#include "optim/OptimizerBase.hpp"
#include <Eigen/Dense>

namespace optim {
class DifferentiableFunction;
class TwiceDifferentiableFunction;
} // namespace optim

namespace optim::logger {
class Logger;
}

namespace optim::linesearch {
class StepLengthPolicy;

template <typename FuncType> class LineSearchBase : public optim::OptimizerBase {
  public:
    LineSearchBase(std::unique_ptr<StepLengthPolicy> step_length_policy, int max_iterations,
                   optim::ConvergenceCriteria criteria,
                   std::shared_ptr<optim::logger::Logger> logger);
    void setLogger(std::shared_ptr<optim::logger::Logger> logger) override;

    virtual optim::OptimizationResult optimize(const FuncType &f, const Eigen::VectorXd &x0) = 0;

  protected:
    std::unique_ptr<StepLengthPolicy> step_length_policy_;
};

extern template class LineSearchBase<optim::DifferentiableFunction>;
extern template class LineSearchBase<optim::TwiceDifferentiableFunction>;

} // namespace optim::linesearch
