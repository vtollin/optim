#pragma once
#include "optim/OptimizerBase.hpp"
#include "optim/OptimizationResult.hpp"
#include <Eigen/Dense>

namespace optim::abstract { class DifferentiableFunction; }

namespace optim::logger {
class Logger;
}

namespace optim::linesearch {
class SearchStrategyBase;
struct ArmijoConfig;
struct WolfeConfig;

enum class SearchStrategy { ARMIJO, STRONG_WOLFE };
class LineSearchBase : public optim::OptimizerBase {
  public:
    LineSearchBase(SearchStrategy strategy, int max_iterations,
                   optim::ConvergenceCriteria criteria, std::shared_ptr<optim::logger::Logger> logger);

    void setStrategy(SearchStrategy s);

    void setConfig(const ArmijoConfig &cfg);
    void setConfig(const WolfeConfig &cfg);
    void setLogger(std::shared_ptr<optim::logger::Logger> logger) override;

    virtual optim::OptimizationResult optimize(const optim::abstract::DifferentiableFunction &f,
                                               const Eigen::VectorXd &x0) = 0;

  protected:
    std::shared_ptr<SearchStrategyBase> search_strategy_;
};
} // namespace optim::linesearch
