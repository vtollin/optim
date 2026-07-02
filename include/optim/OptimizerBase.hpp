#pragma once
#include "optim/IterationObserver.hpp"
#include "optim/OptimizationResult.hpp"
#include <Eigen/Dense>
#include <memory>
#include <optional>
#include <vector>

namespace optim::logger {
class Logger;
}

namespace optim {
struct ConvergenceCriteria {
    double grad_tol = 1e-8;
    double step_tol = 1e-8;
    std::optional<double> f_tol;
};
class OptimizerBase {
  public:
    virtual ~OptimizerBase() = default;
    // Sets logger. Does NOT take ownership: the caller owns the logger and must keep it alive
    // for as long as this optimizer (or any optimizer sharing it) is in use. Pass nullptr to
    // disable logging.
    virtual void setLogger(optim::logger::Logger *logger);

    // Sets observer used for iteration tracing. Does NOT take ownership: the caller owns
    // the observer and must keep it alive for as long as this optimizer is in use.
    void addObserver(optim::trace::IterationObserver *obs) {
        if (obs)
            observers_.push_back(obs);
    }

  protected:
    // 'logger' is observed, not owned. See setLogger for the lifetime contract.
    OptimizerBase(int max_iterations, ConvergenceCriteria criteria, optim::logger::Logger *logger);

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
    optim::logger::Logger *logger_;

  private:
    std::vector<optim::trace::IterationObserver *> observers_;
};
} // namespace optim
