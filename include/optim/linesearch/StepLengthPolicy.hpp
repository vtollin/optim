#pragma once
#include "optim/linesearch/StepStatus.hpp"
#include <Eigen/Dense>

namespace optim {
class DifferentiableFunction;
}

namespace optim::linesearch {

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

  protected:
    StepLengthPolicy() = default;
};
} // namespace optim::linesearch
