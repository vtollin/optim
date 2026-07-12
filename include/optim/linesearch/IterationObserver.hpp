#pragma once
#include "optim/linesearch/StepStatus.hpp"
#include <Eigen/Dense>

namespace optim::linesearch {
struct IterationInfo {
    int iter;
    double fval;
    Eigen::VectorXd x, grad, step;
    StepStatus step_status;
};

class IterationObserver {
  public:
    virtual ~IterationObserver() = default;
    virtual void onIteration(const IterationInfo &info) = 0;

    // Lifecycle hooks; default no-ops. Allows a file/CSV observer
    // to print a header on start or flush/close on finish
    virtual void onStart() {}
    virtual void onFinish() {}
};

} // namespace optim::linesearch