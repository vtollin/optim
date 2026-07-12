#pragma once
#include "optim/OptimizationResult.hpp"
#include "optim/OptimizerBase.hpp"
#include "optim/linesearch/IterationObserver.hpp"
#include "optim/linesearch/StepLengthPolicy.hpp"
#include <Eigen/Dense>
#include <memory>

namespace optim {
class DifferentiableFunction;
class TwiceDifferentiableFunction;
} // namespace optim

namespace optim::linesearch {

template <typename FuncType> class LineSearchBase : public optim::OptimizerBase {
  public:
    optim::OptimizationResult optimize(const FuncType &f, const Eigen::VectorXd &x0);

  protected:
    LineSearchBase(std::unique_ptr<StepLengthPolicy> step_length_policy, int max_iterations,
                   optim::ConvergenceCriteria criteria);
    std::unique_ptr<StepLengthPolicy> step_length_policy_;

  private:
    std::vector<optim::linesearch::IterationObserver *> observers_;
    void notifyStart() {
        for (auto *o : observers_)
            o->onStart();
    }
    void notifyFinish() {
        for (auto *o : observers_)
            o->onFinish();
    }
    void notifyIteration(const optim::linesearch::IterationInfo &info) {
        for (auto *o : observers_)
            o->onIteration(info);
    }

    virtual Eigen::VectorXd computeDirection(const FuncType &f, const Eigen::VectorXd &x,
                                             const Eigen::VectorXd &grad) = 0;
    virtual void resetState(int n) {}
    virtual void updateState(const Eigen::VectorXd &s, const Eigen::VectorXd &y) {}
};

extern template class LineSearchBase<optim::DifferentiableFunction>;
extern template class LineSearchBase<optim::TwiceDifferentiableFunction>;

} // namespace optim::linesearch
