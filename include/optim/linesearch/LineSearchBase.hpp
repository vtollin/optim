#pragma once
#include "optim/OptimizationResult.hpp"
#include "optim/OptimizerBase.hpp"
#include "optim/linesearch/StepLengthPolicy.hpp"

#include <Eigen/Dense>

namespace optim {
class DifferentiableFunction;
class TwiceDifferentiableFunction;
} // namespace optim

namespace optim::linesearch {

template <typename FuncType> class LineSearchBase : public optim::OptimizerBase {
  public:
    LineSearchBase(std::unique_ptr<StepLengthPolicy> step_length_policy, int max_iterations,
                   optim::ConvergenceCriteria criteria);

    optim::OptimizationResult optimize(const FuncType &f, const Eigen::VectorXd &x0);

  protected:
    std::unique_ptr<StepLengthPolicy> step_length_policy_;

  private:
    virtual Eigen::VectorXd computeDirection(const FuncType &f, const Eigen::VectorXd &x,
                                             const Eigen::VectorXd &grad) = 0;
    virtual void resetState(int n) {}
    virtual void updateState(const Eigen::VectorXd &s, const Eigen::VectorXd &y) {}
};

extern template class LineSearchBase<optim::DifferentiableFunction>;
extern template class LineSearchBase<optim::TwiceDifferentiableFunction>;

} // namespace optim::linesearch
