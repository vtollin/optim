#include "optim/diagnostics/HessianChecker.hpp"
#include "optim/Functions.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <limits>

namespace optim::diagnostics {

// Perturbation h ~ cbrt(u): central difference balances O(h^2) truncation error against O(u/h)
// roundoff error, minimized at h proportional to u^(1/3). Computes numerical hessian in one pass,
// symmetrizes it, and then compares to analytic hessian() in another.
HessianCheckResult hessian_check(const optim::TwiceDifferentiableFunction &f,
                                 const Eigen::VectorXd &x, double rel_tol, double abs_tol) {

    Eigen::Index n = x.size();
    Eigen::MatrixXd n_hess(n, n);
    double perturbation = std::cbrt(std::numeric_limits<double>::epsilon() / 2.0);
    for (Eigen::Index j = 0; j < n; ++j) {
        double h = perturbation * std::max(std::abs(x(j)), 1.0);
        Eigen::VectorXd e_j = Eigen::VectorXd::Unit(n, j);
        Eigen::VectorXd forward = f.gradient(x + h * e_j);
        Eigen::VectorXd backward = f.gradient(x - h * e_j);
        n_hess.col(j) = (forward - backward) / (2.0 * h);
    }

    Eigen::MatrixXd symm_n_hess = (n_hess + n_hess.transpose()).eval() / 2.0;
    Eigen::MatrixXd a_hess = f.hessian(x);
    double max_rel_error = -1;
    double max_abs_error = -1;
    Eigen::Index worst_row = -1;
    Eigen::Index worst_col = -1;
    bool result = true;
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < n; ++j) {
            double abs_error = std::abs(symm_n_hess(i, j) - a_hess(i, j));
            double rel_error = abs_error / std::max(std::abs(a_hess(i, j)), abs_tol);

            if (rel_error > max_rel_error) {
                max_rel_error = rel_error;
                worst_row = i;
                worst_col = j;
            }
            if (abs_error > max_abs_error) {
                max_abs_error = abs_error;
            }
            if (rel_error > rel_tol || abs_error > abs_tol) {
                result = false;
            }
        }
    }

    double asymmetry = (a_hess - a_hess.transpose()).norm();
    double scale = a_hess.norm();
    bool analytic_symmetric = asymmetry <= 1e-10 * std::max(scale, 1.0);

    return HessianCheckResult{result,        analytic_symmetric, max_abs_error,
                              max_rel_error, worst_row,          worst_col};
}
} // namespace optim::diagnostics