#pragma once
#include <Eigen/Dense>
#include <string>

namespace optim {

enum class StopReason {
    GRADIENT_CONVERGED,
    MAX_ITERS_REACHED,
    STEP_STALLED,
    F_CHANGE_BELOW_TOL,
    LINE_SEARCH_FAILURE
};

struct OptimizationResult {
    Eigen::VectorXd x_opt;
    double f_val;
    int iterations;
    bool converged;
    StopReason reason;
};
} // namespace optim