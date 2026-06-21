#pragma once
#include <Eigen/Dense>
#include <memory>

namespace optim::logger {
class Logger;
}

namespace optim { class DifferentiableFunction; }

namespace optim::linesearch {
class StepLengthPolicy {
  public:
    virtual ~StepLengthPolicy() = default;

    virtual double computeStep(const optim::DifferentiableFunction &f,
                               const Eigen::VectorXd &x,
                               const Eigen::VectorXd &direction,
                               const Eigen::VectorXd &gradient) = 0;

    void setLogger(std::shared_ptr<optim::logger::Logger> logger) { logger_ = logger; }

  protected:
    explicit StepLengthPolicy(std::shared_ptr<optim::logger::Logger> logger) : logger_(logger) {}
    std::shared_ptr<optim::logger::Logger> logger_;
};
} // namespace optim::linesearch
