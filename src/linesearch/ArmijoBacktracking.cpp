#include "optim/linesearch/ArmijoBacktracking.hpp"
#include "optim/linesearch/Interpolation.hpp"
#include "optim/logger/Logger.hpp"
#include "optim/Functions.hpp"
#include "optim/OptimizerUtility.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <optional>

using namespace optim::linesearch;
using optim::DifferentiableFunction;

ArmijoBacktracking::ArmijoBacktracking(const ArmijoConfig &config,
                                       optim::logger::Logger *logger)
    : StepLengthPolicy(logger), config_(config) {
    if (config.alpha_init <= 0.0) {
        throw std::invalid_argument("[ArmijoBacktracking] alpha_init must be > 0.");
    }
    if (config.c1 <= 0.0 || config.c1 >= 1.0) {
        throw std::invalid_argument("[ArmijoBacktracking] c1 must be in (0, 1).");
    }
    if (config.max_iters <= 0) {
        throw std::invalid_argument("[ArmijoBacktracking] max_iters must be positive.");
    }
}

// Shrinks alpha from alpha_init until the Armijo sufficient decrease condition is met.
// Proposes each trial step via polynomial interpolation, falling back to bisection when
// the interpolated step is not sufficiently conservative. N&W Algorithm 3.1, pp. 56-58.
double ArmijoBacktracking::computeStep(const DifferentiableFunction &f, const Eigen::VectorXd &x,
                                       const Eigen::VectorXd &direction,
                                       const Eigen::VectorXd &gradient) {
    double dir_deriv = gradient.dot(direction);
    if (dir_deriv >= 0.0) {
        throw std::runtime_error("[ArmijoBacktracking] direction is not a descent direction.");
    }
    double alpha_min = optim::utility::sqrt_epsilon * (1.0 + x.norm());
    double alpha = config_.alpha_init;
    if (alpha < alpha_min) { // alpha_init already at machine-precision floor for this x
        if (logger_ && logger_->shouldLog(optim::logger::Verbosity::WARN)) {
            logger_->log("[ArmijoBacktracking] Warning: alpha_init (" + std::to_string(alpha) +
                         ") below minimum step length (" + std::to_string(alpha_min) +
                         ") at this x. Returning 0.");
        }
        return 0.0;
    }

    double f_x = f.evaluate(x);
    double phi = f.evaluate(x + alpha * direction);

    double best_alpha = alpha;
    double best_phi = phi;
    std::optional<double> alpha_prev = std::nullopt;
    std::optional<double> phi_prev = std::nullopt;
    int k = 0;
    for (; k < config_.max_iters; ++k) {
        if (phi <= f_x + config_.c1 * alpha * dir_deriv) { // sufficient decrease (N&W eq. 3.4)
            return alpha;
        }
        double new_alpha =
            nextTrialStep(alpha, phi, alpha_prev, phi_prev, f_x, dir_deriv, alpha_min);
        if (new_alpha < alpha_min) {
            if (logger_ && logger_->shouldLog(optim::logger::Verbosity::WARN)) {
                logger_->log("[ArmijoBacktracking] Warning: Alpha fell below alpha_min. Returning "
                             "best step found.");
            }
            break; // if step falls below minimum, safeguards against machine precision errors
        }
        alpha_prev = alpha;
        phi_prev = phi;
        phi = f.evaluate(x + new_alpha * direction);
        alpha = new_alpha;
        if (phi < best_phi) {
            best_alpha = new_alpha;
            best_phi = phi;
        }
        double tol = optim::utility::epsilon * (std::max(std::abs(phi_prev.value()), std::abs(phi)) + 1.0);
        if (std::abs(phi - phi_prev.value()) < tol) { // relative tolerance with absolute floor
            if (logger_ && logger_->shouldLog(optim::logger::Verbosity::WARN)) {
                logger_->log("[ArmijoBacktracking] Warning: Change in phi fell below machine "
                             "precision. Returning best step found.");
            }
            break;
        }
    }
    if (logger_ && logger_->shouldLog(optim::logger::Verbosity::INFO)) {
        logger_->log("[ArmijoBacktracking] Completed in " + std::to_string(k) +
                     " iterations, final alpha = " + std::to_string(best_alpha));
    }
    return best_alpha;
}

// Quadratic interpolation on the first call; cubic once a previous iterate is available.
// Bisects if the interpolated step does not reduce alpha by at least 10%. N&W pp. 56-57.
double ArmijoBacktracking::nextTrialStep(double alpha, double phi, std::optional<double> alpha_prev,
                                         std::optional<double> phi_prev, double phi0,
                                         double phi_prime0, double alpha_min) {
    double alpha_new;
    if (!alpha_prev.has_value()) {
        alpha_new = interpolation::quadraticMinimizer(0.0, alpha, phi0, phi, phi_prime0);
    } else {
        alpha_new = interpolation::cubicMinimizer(phi0, phi_prev.value(), phi, phi_prime0,
                                                  alpha_prev.value(), alpha);
    }
    if (alpha - alpha_new <= 0.1 * alpha || alpha_new <= alpha_min) {
        alpha_new = alpha / 2; // bisect: interpolated step not a sufficient reduction
    }
    return alpha_new;
}
