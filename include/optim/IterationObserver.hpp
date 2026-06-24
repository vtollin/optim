#pragma once
#include <Eigen/Dense>
#include <optional>

namespace optim::trace {
struct IterationInfo {
    int iter;
    double fval;
    Eigen::VectorXd x, grad, step;
    // Trust region only; absent for line search optimizers
    std::optional<double> rho;
    std::optional<double> delta;
    std::optional<bool> accepted;
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

} // namespace optim::trace