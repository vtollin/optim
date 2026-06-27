#include "optim/diagnostics/GradientChecker.hpp"
#include "optim/Functions.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <limits>

namespace optim::diagnostics {

// Perturbation h ~ cbrt(u): central difference balances O(h^2) truncation error against O(u/h)
// roundoff error, minimized at h proportional to u^(1/3).
GradientCheckResult gradient_check(const optim::DifferentiableFunction &f, const Eigen::VectorXd &x,
                                   double rel_tol, double abs_tol) {
    double perturbation = std::cbrt(std::numeric_limits<double>::epsilon() / 2.0);
    Eigen::VectorXd analytic_grad = f.gradient(x);
    double max_abs_error = -1;
    double max_rel_error = -1;
    Eigen::Index worst_coord = -1;
    bool result = true;
    for (Eigen::Index i = 0; i < x.size(); ++i) {
        double h = perturbation * std::max(std::abs(x(i)), 1.0);
        Eigen::VectorXd e_i = Eigen::VectorXd::Unit(x.size(), i);
        double forward = f.evaluate(x + h * e_i);
        double backward = f.evaluate(x - h * e_i);
        double partial_i = (forward - backward) / (2 * h);
        double abs_error = std::abs(analytic_grad(i) - partial_i);
        double rel_error = abs_error / std::max(std::abs(analytic_grad(i)), abs_tol);

        if (rel_error > max_rel_error) {
            max_rel_error = rel_error;
            worst_coord = i;
        }
        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
        }
        if (rel_error > rel_tol || abs_error > abs_tol) {
            result = false;
        }
    }
    return GradientCheckResult{result, max_abs_error, max_rel_error, worst_coord};
}
} // namespace optim::diagnostics