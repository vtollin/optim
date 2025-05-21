#include "optimization/LineSearch/ArmijoBacktracking.hpp"
#include "optimization/ObjectiveFunctionBase.hpp"
#include "optimization/OptimizationUtils.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <limits>

using namespace LineSearch;

ArmijoBacktracking::ArmijoBacktracking(const ArmijoConfig &config) : config_(config) {
    if (config.alpha_init <= 0.0) {
        throw std::invalid_argument("ArmijoConfig: alpha_init must be > 0.");
    }
    if (config.rho <= 0.0 || config.rho >= 1.0) {
        throw std::invalid_argument("ArmijoConfig: rho must be in (0, 1).");
    }
    if (config.c <= 0.0 || config.c >= 1.0) {
        throw std::invalid_argument("ArmijoConfig: c must be in (0, 1).");
    }
    if (config.max_iters <= 0) {
        throw std::invalid_argument("ArmijoConfig: max_iters must be positive.");
    }
}

Eigen::VectorXd ArmijoBacktracking::computeStep(const ObjectiveFunctionBase &f,
                                                const Eigen::VectorXd &x,
                                                const Eigen::VectorXd &direction,
                                                const Eigen::VectorXd &gradient,
                                                double alpha_override) {
    double alpha_min = Utility::sqrt_epsilon *
                       (1.0 + x.norm()); // minimum protects against machine precision errors
    double alpha = (alpha_override > 0.0) ? alpha_override : config_.alpha_init;
    double new_alpha;
    double alpha_prev = -1.0; // invalid value for first iteration
    double f_x = f.evaluate(x);
    double phi_prev;
    double dir_deriv = Utility::directionalDerivative(f, x, direction);
    if (dir_deriv >= 0.0) {
        throw std::runtime_error("Backtracking line search: direction is not a descent direction");
    }
    double phi = f.evaluate(x + alpha * direction);
    double best_phi = phi;
    double best_alpha = alpha;
    for (int k = 0; k < config_.max_iters; ++k) {
        if (phi <= f_x + config_.c * alpha * dir_deriv) {
            return alpha * direction;
        }
        double new_alpha =
            nextTrialStep(alpha, phi, f_x, dir_deriv, alpha_prev, phi_prev, alpha_min);
        if (new_alpha < alpha_min) {
            if (config_.isVerbose) {
                std::cerr
                    << "[ArmijoBacktracking] Warning: Alpha fell below alpha_min. Returning best "
                       "step found.\n";
            }
            break; // if step falls below minimum, safeguards against machine precision errors
        }
        alpha_prev = alpha;
        phi_prev = phi;
        phi = f.evaluate(x + new_alpha * direction);
        alpha = new_alpha;
        double tol_rel = Utility::epsilon * (std::max(std::abs(phi_prev), std::abs(phi)) + 1.0);
        if (std::abs(phi - phi_prev) < tol_rel) { // relative tolerance with absolute floor
            if (config_.isVerbose) {
                std::cerr << "[ArmijoBacktracking] Warning: Change in phi fell below machine "
                             "precision. Returning best step found.\n";
            }
            break; // if no detectable change in phi after step
        }
        if (phi < best_phi) {
            best_alpha = new_alpha;
            best_phi = phi;
        }
    }
    return best_alpha * direction; // best alpha on stall
}

double ArmijoBacktracking::nextTrialStep(double alpha, double phi, double phi0, double phi_prime0,
                                         double alpha_prev, double phi_prev, double alpha_min) {
    double alpha_new;
    switch (config_.strategy) {
    case ArmijoConfig::TrialStepOpts::GEOMETRIC:
        alpha_new = config_.rho * alpha;
    case ArmijoConfig::TrialStepOpts::GUARDED_INTERPOLATION:
        if (alpha_prev < 0) { // if first iteration
            alpha_new = quadraticInterpolation(phi0, phi, phi_prime0, alpha);
        } else {
            alpha_new = cubicInterpolation(phi0, phi_prev, phi, phi_prime0, alpha_prev, alpha);
        }
        if (alpha - alpha_new <= 0.1 * alpha || alpha_new <= alpha_min) { // safeguard
            alpha_new = alpha / 2;                                        // fall back to bisection
        }
    }
    return alpha_new;
}
double ArmijoBacktracking::quadraticInterpolation(double endpoint_lo, double endpoint_hi,
                                                  double deriv, double alpha) {
    // return minimizer
    double num = deriv * alpha * alpha;
    double denom = 2 * (endpoint_hi - endpoint_lo - deriv * alpha);
    return -num / denom;
}

double ArmijoBacktracking::cubicInterpolation(double endpoint_lo, double endpoint_hi,
                                              double midpoint, double deriv, double alpha_prev,
                                              double alpha) {
    double a_num = alpha_prev * alpha_prev * (midpoint - endpoint_lo - deriv * alpha) -
                   alpha * alpha * (endpoint_hi - endpoint_lo - deriv * alpha_prev);
    double b_num =
        -alpha_prev * alpha_prev * alpha_prev * (midpoint - endpoint_lo - deriv * alpha) +
        alpha * alpha * alpha * (endpoint_hi - endpoint_lo - deriv * alpha_prev);
    double denom = alpha_prev * alpha_prev * alpha * alpha * (alpha - alpha_prev);
    double a = a_num / denom;
    double b = b_num / denom;
    return (-b + std::sqrt(b * b - 3 * a * deriv)) / 3 * a;
}