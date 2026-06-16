#pragma once
#include "optim/linesearch/LineSearchBase.hpp"
#include "optim/OptimizationResult.hpp"
#include <Eigen/Dense>

namespace optim::logger {
class Logger;
}

namespace optim::linesearch {
class SearchStrategyBase;
}

namespace optim::abstract {
class DifferentiableFunction;
class TwiceDifferentiableFunction;
}

namespace optim::linesearch {
struct CholeskyFactor {
    Eigen::MatrixXd L;
    Eigen::VectorXd d;
};
class Newton : public LineSearchBase {
  public:
    Newton(SearchStrategy search_strategy = SearchStrategy::ARMIJO, int max_iterations = 1000,
           optim::ConvergenceCriteria criteria = {},
           std::shared_ptr<optim::logger::Logger> logger = nullptr);

    // Primary API: compile-time type check for direct callers
    optim::OptimizationResult optimize(const optim::abstract::TwiceDifferentiableFunction &f,
                                       const Eigen::VectorXd &x0);

    // LineSearchBase override: dynamic_casts to TwiceDifferentiableFunction for polymorphic use
    optim::OptimizationResult optimize(const optim::abstract::DifferentiableFunction &f,
                                       const Eigen::VectorXd &x0) override;

  private:
    CholeskyFactor modifiedCholesky(const Eigen::MatrixXd &hessian);
};
} // namespace optim::linesearch
