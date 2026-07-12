#pragma once
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

  protected:
    OptimizerBase(int max_iterations, ConvergenceCriteria criteria);

    int max_iterations_;
    ConvergenceCriteria criteria_;
};
} // namespace optim
