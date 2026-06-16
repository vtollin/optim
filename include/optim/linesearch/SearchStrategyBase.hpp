#pragma once
#include <Eigen/Dense>
#include <memory>

namespace optim::logger {
class Logger;
}

namespace optim::abstract { class DifferentiableFunction; }

namespace optim::linesearch {
class SearchStrategyBase {
  public:
    virtual ~SearchStrategyBase() = default;

    virtual double computeStep(const optim::abstract::DifferentiableFunction &f,
                               const Eigen::VectorXd &x,
                               const Eigen::VectorXd &direction,
                               const Eigen::VectorXd &gradient) = 0;

    void setLogger(std::shared_ptr<optim::logger::Logger> logger) { logger_ = logger; }

  protected:
    explicit SearchStrategyBase(std::shared_ptr<optim::logger::Logger> logger) : logger_(logger) {}
    std::shared_ptr<optim::logger::Logger> logger_;
};
} // namespace optim::linesearch
