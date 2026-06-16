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
    virtual void setLogger(std::shared_ptr<optim::logger::Logger> logger);

  protected:
    OptimizerBase(int max_iterations, ConvergenceCriteria criteria,
                  std::shared_ptr<optim::logger::Logger> logger);
    int max_iterations_;
    ConvergenceCriteria criteria_;
    std::shared_ptr<optim::logger::Logger> logger_;
};
} // namespace optim
