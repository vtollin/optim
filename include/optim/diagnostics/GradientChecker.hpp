#pragma once
#include "optim/Functions.hpp"
#include <Eigen/Dense>

namespace optim::diagnostics {
struct GradientCheckResult {
    bool result;
    double max_abs_error;
    double max_rel_error;
    Eigen::Index worst_coord;
};

// Verifies f's analytic gradient() against a central-difference approximation
// computed from evaluate() alone. Returns aggregated error info and a
// pass/fail flag based on rel_tol / abs_tol.
GradientCheckResult gradient_check(const DifferentiableFunction &f, const Eigen::VectorXd &x,
                                   double rel_tol = 1e-6, double abs_tol = 1e-8);
} // namespace optim::diagnostics