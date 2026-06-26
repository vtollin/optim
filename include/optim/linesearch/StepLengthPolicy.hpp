#pragma once
#include <Eigen/Dense>
#include <memory>

namespace optim::logger {
class Logger;
}

namespace optim {
class DifferentiableFunction;
}

namespace optim::linesearch {

enum class StepStatus { SUCCESS, INADEQUATE, FAILURE };
struct StepResult {
    double alpha;
    StepStatus status;
};
class StepLengthPolicy {
  public:
    virtual ~StepLengthPolicy() = default;

    virtual StepResult computeStep(const optim::DifferentiableFunction &f, const Eigen::VectorXd &x,
                                   const Eigen::VectorXd &direction,
                                   const Eigen::VectorXd &gradient) = 0;

    void setLogger(optim::logger::Logger *logger) { logger_ = logger; }

  protected:
    explicit StepLengthPolicy(optim::logger::Logger *logger) : logger_(logger) {}
    optim::logger::Logger *logger_;
};
} // namespace optim::linesearch
