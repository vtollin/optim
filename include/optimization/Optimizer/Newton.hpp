#pragma once
#include "OptimizationResult.hpp"
#include "optimization/Optimizer/OptimizerBase.hpp"
#include <Eigen/Dense>

namespace LineSearch {
class SearchStrategyBase;
}

class Logger;
class ObjectiveFunctionBase;

namespace Optimizer {
class Newton : public OptimizerBase {
  public:
    Newton(std::shared_ptr<LineSearch::SearchStrategyBase> search_strategy,
           int max_iterations_ = 10000, double tol_ = 1e-6,
           std::shared_ptr<Logger> logger = nullptr);

    OptimizationResult optimize(const ObjectiveFunctionBase &f, const Eigen::VectorXd &x0) override;

  private:
    std::shared_ptr<LineSearch::SearchStrategyBase> search_strategy_;
    int max_iterations_;
    double tol_;
    std::shared_ptr<Logger> logger_;
};
} // namespace Optimizer