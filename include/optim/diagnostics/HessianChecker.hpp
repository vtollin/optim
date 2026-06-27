#pragma once
#include "optim/Functions.hpp"
#include <Eigen/Dense>

namespace optim::diagnostics {
struct HessianCheckResult {
    bool result;
    bool analytic_symmetric;
    double max_abs_error;
    double max_rel_error;
    Eigen::Index worst_row;
    Eigen::Index worst_col;
};

// Verifies f's analytic hessian() against a central-difference approximation computed from
// gradient() alone. Returns aggregate error info, a pass/fail flag, and a flag indicating whether
// the analytic hessian is symmetric.
HessianCheckResult hessian_check(const TwiceDifferentiableFunction &f, const Eigen::VectorXd &x,
                                 double rel_tol = 1e-6, double abs_tol = 1e-8);
} // namespace optim::diagnostics