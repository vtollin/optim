#include "optimization/Backtracking.hpp"
#include "optimization/ObjectiveFunction.hpp"
#include "optimization/OptimizationUtils.hpp"

Backtracking::Backtracking(double alpha_init, double rho, double c, double alpha_min)
    : alpha_init_(alpha_init), rho_(rho), c_(c), alpha_min_(alpha_min) {
    if (alpha_init <= 0.0 || rho <= 0.0 || rho >= 1.0 || c <= 0.0 || c >= 1.0 || alpha_min <= 0.0) {
        throw std::invalid_argument("Invalid backtracking line search parameters.");
    }
}

double Backtracking::chooseStep(ObjectiveFunction &f, const Eigen::VectorXd &x,
                                const Eigen::VectorXd &direction, const Eigen::VectorXd &gradient,
                                double alpha_override) {
    double alpha = (alpha_override > 0.0) ? alpha_override : alpha_init_;

    double f_x = f.evaluate(x);
    double dir_deriv = directionalDerivative(f, x, direction);

    if (dir_deriv >= 0.0) {
        throw std::runtime_error("Backtracking line search: direction is not a descent direction");
    }
    for (int k = 0; k < max_iters_; ++k) {
        if (f.evaluate(x + alpha * direction) <= f_x + c_ * alpha * dir_deriv) {
            return alpha;
        }

        alpha *= rho_;

        if (alpha < alpha_min_) {
            throw std::runtime_error(
                "Backtracking line search failed: alpha fell below alpha_min.");
        }
    }
    throw std::runtime_error("Backtracking line search failed: max iterations reached");
}
