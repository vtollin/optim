#pragma once
#include "optim/trace/IterationObserver.hpp"
#include "optim/OptimizationResult.hpp"
#include <Eigen/Dense>
#include <optional>
#include <vector>

namespace optim {
struct ConvergenceCriteria {
    double grad_tol = 1e-8;
    double step_tol = 1e-8;
    std::optional<double> f_tol;
};
class OptimizerBase {
  public:
    virtual ~OptimizerBase() = default;

    // Sets observer used for iteration tracing. Does NOT take ownership: the caller owns
    // the observer and must keep it alive for as long as this optimizer is in use.
    void addObserver(optim::trace::IterationObserver *obs) {
        if (obs)
            observers_.push_back(obs);
    }

  protected:
    OptimizerBase(int max_iterations, ConvergenceCriteria criteria);

    void notifyStart() {
        for (auto *o : observers_)
            o->onStart();
    }
    void notifyFinish() {
        for (auto *o : observers_)
            o->onFinish();
    }
    void notifyIteration(const optim::trace::IterationInfo &info) {
        for (auto *o : observers_)
            o->onIteration(info);
    }
    int max_iterations_;
    ConvergenceCriteria criteria_;

  private:
    std::vector<optim::trace::IterationObserver *> observers_;
};
} // namespace optim
