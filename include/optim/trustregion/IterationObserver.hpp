#pragma once
#include "optim/trustregion/SubproblemStatus.hpp"
#include <Eigen/Dense>

namespace optim::trustregion {
struct IterationInfo {
    int iter;
    double fval, rho, delta;
    Eigen::VectorXd x, grad, step;
    SubproblemStatus status;
    bool accepted;
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

} // namespace optim::trustregion