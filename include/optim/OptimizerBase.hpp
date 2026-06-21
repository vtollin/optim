#pragma once
#include "optim/OptimizationResult.hpp"
#include <Eigen/Dense>
#include <memory>

namespace optim::logger {
class Logger;
}

namespace optim {
struct ConvergenceCriteria {
    double grad_tol = 1e-8;
    double step_tol = 1e-8;
    double f_tol = 1e-8;
};
class OptimizerBase {
  public:
    virtual ~OptimizerBase() = default;
    // Sets the logger used for iteration tracing. Does NOT take ownership:
    // the caller owns the logger and must keep it alive for as long as this
    // optimizer (or any optimizer sharing it) is in use. Pass nullptr to
    // disable logging.
    virtual void setLogger(optim::logger::Logger *logger);

  protected:
    // 'logger' is observed, not owned. See setLogger for the lifetime contract.
    OptimizerBase(int max_iterations, ConvergenceCriteria criteria, optim::logger::Logger *logger);
    int max_iterations_;
    ConvergenceCriteria criteria_;
    optim::logger::Logger *logger_;
};
} // namespace optim
