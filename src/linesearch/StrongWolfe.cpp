#include "optim/linesearch/StrongWolfe.hpp"
#include "optim/linesearch/Interpolation.hpp"
#include "optim/logger/Logger.hpp"
#include "optim/Functions.hpp"
#include "optim/OptimizerUtility.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <limits>

using namespace optim::linesearch;
using optim::DifferentiableFunction;

StrongWolfe::StrongWolfe(const WolfeConfig &config, optim::logger::Logger *logger)
    : StepLengthPolicy(logger), config_(config) {
    if (config.alpha_init <= 0.0) {
        throw std::invalid_argument("[StrongWolfe] alpha_init must be > 0.");
    }
    if (config.alpha_max <= 0.0) {
        throw std::invalid_argument("[StrongWolfe] alpha_max must be > 0.");
    }
    if (config.alpha_init > config.alpha_max) {
        throw std::invalid_argument("[StrongWolfe] alpha_init must not exceed alpha_max.");
    }
    if (config.max_iters <= 0) {
        throw std::invalid_argument("[StrongWolfe] max_iters must be positive.");
    }
    if (config.rho <= 1.0) {
        throw std::invalid_argument("[StrongWolfe] rho must be > 1.0 for forward step.");
    }
    if (config.c1 <= 0.0 || config.c1 >= 1.0) {
        throw std::invalid_argument("[StrongWolfe] c1 must be in (0, 1).");
    }
    if (config.c2 <= config.c1 || config.c2 >= 1.0) {
        throw std::invalid_argument("[StrongWolfe] c2 must be in (c1, 1).");
    }
}

// Expands alpha geometrically from alpha_init until the Strong Wolfe conditions are satisfied
// directly or a bracket containing a Wolfe point is found. Delegates to zoom() once a bracket
// is identified. N&W Algorithm 3.5, pp. 59-62.
double StrongWolfe::computeStep(const DifferentiableFunction &f, const Eigen::VectorXd &x,
                                const Eigen::VectorXd &direction, const Eigen::VectorXd &gradient) {
    double phi0 = f.evaluate(x);
    double phi_prime0 = gradient.dot(direction);
    double alpha = config_.alpha_init;
    double phi = f.evaluate(x + alpha * direction);

    double best_alpha = alpha;
    double best_phi = phi;
    double alpha_prev = 0.0;
    double phi_prev =
        std::numeric_limits<double>::max(); // sentinel: no previous point on first iteration
    int stall_counter = 0;
    for (int k = 0; k < config_.max_iters; ++k) {
        if (phi > phi0 + config_.c1 * alpha * phi_prime0 ||
            phi > phi_prev) { // bracket found: Armijo failed or phi rose
            return zoom(alpha_prev, alpha, f, x, direction, phi0, phi_prime0);
        }
        double phi_prime = optim::utility::directionalDerivative(f, x + alpha * direction, direction);
        if (std::abs(phi_prime) <=
            -config_.c2 * phi_prime0) { // strong Wolfe conditions satisfied (N&W eq. 3.7)
            return alpha;
        }
        if (phi_prime >= 0) { // phi' flipped positive: bracket straddles a minimum
            return zoom(alpha, alpha_prev, f, x, direction, phi0, phi_prime0);
        }
        double alpha_next = alpha * config_.rho;
        if (alpha_next > config_.alpha_max) {
            if (logger_ && logger_->shouldLog(optim::logger::Verbosity::WARN)) {
                logger_->log("[StrongWolfe] Warning: Alpha exceeded alpha_max: returning best step "
                             "found.");
            }
            break;
        }
        alpha_prev = alpha;
        phi_prev = phi;
        phi = f.evaluate(x + alpha_next * direction);
        if (phi < best_phi) {
            best_alpha = alpha_next;
            best_phi = phi;
        }
        double tol = optim::utility::sqrt_epsilon * (std::max(std::abs(phi), std::abs(phi_prev)) + 1.0);
        if (std::abs(phi - phi_prev) < tol) { // relative tolerance with absolute floor
            ++stall_counter;
            if (stall_counter == 5) {
                if (logger_ && logger_->shouldLog(optim::logger::Verbosity::WARN)) {
                    logger_->log("[StrongWolfe] Warning: Change in phi fell below machine "
                                 "precision for 5 iterations. Returning best step found.");
                }
                break;
            }
        } else {
            stall_counter = 0;
        }
        alpha = alpha_next;
    }
    return best_alpha;
}

// Refines a bracket [alpha_lo, alpha_hi] known to contain a Strong Wolfe point until one is
// found or the bracket collapses to machine precision. Proposes each trial via Hermite cubic
// interpolation with quadratic and bisection fallbacks. N&W Algorithm 3.6, pp. 60-61.
double StrongWolfe::zoom(double alpha_lo, double alpha_hi, const DifferentiableFunction &f,
                         const Eigen::VectorXd &x, const Eigen::VectorXd &direction, double phi0,
                         double phi_prime0) {
    double phi_lo = f.evaluate(x + alpha_lo * direction);
    double phi_hi = f.evaluate(x + alpha_hi * direction);
    double phi_prime_lo = optim::utility::directionalDerivative(f, x + alpha_lo * direction, direction);
    double phi_prime_hi = optim::utility::directionalDerivative(f, x + alpha_hi * direction, direction);
    for (int i = 0; i < config_.zoom_max_iters; ++i) {
        double alpha = interpolation::cubicHermiteMinimizer(alpha_lo, alpha_hi, phi_lo, phi_hi,
                                                            phi_prime_lo, phi_prime_hi);
        if (isInvalid(alpha_lo, alpha_hi, alpha)) { // fall back to quadratic
            alpha =
                interpolation::quadraticMinimizer(alpha_lo, alpha_hi, phi_lo, phi_hi, phi_prime_lo);
        }
        if (isInvalid(alpha_lo, alpha_hi, alpha)) { // fall back to bisection
            alpha = 0.5 * (alpha_hi + alpha_lo);
        }
        double phi = f.evaluate(x + alpha * direction);
        if (phi > phi0 + config_.c1 * alpha * phi_prime0 ||
            phi >= phi_lo) { // Armijo failed or phi worsened: update hi
            alpha_hi = alpha;
            phi_hi = phi;
            phi_prime_hi = optim::utility::directionalDerivative(f, x + alpha * direction, direction);
        } else {
            double phi_prime = optim::utility::directionalDerivative(f, x + alpha * direction, direction);
            if (std::abs(phi_prime) <=
                -config_.c2 * phi_prime0) { // strong Wolfe conditions satisfied (N&W eq. 3.7)
                return alpha;
            }
            if (phi_prime * (alpha_hi - alpha_lo) >=
                0) { // phi' points toward hi: hi must move to old lo
                alpha_hi = alpha_lo;
                phi_hi = phi_lo;
                phi_prime_hi = phi_prime_lo;
            }
            alpha_lo = alpha;
            phi_lo = phi;
            phi_prime_lo = phi_prime;
        }
        double tol = optim::utility::sqrt_epsilon * (std::max(alpha_lo, alpha_hi) + 1.0);
        if (std::abs(alpha_hi - alpha_lo) < tol) { // break if bracket falls below tolerance
            break;
        }
    }
    if (logger_ && logger_->shouldLog(optim::logger::Verbosity::WARN)) {
        logger_->log("[StrongWolfe] Warning: zoom() failed to converge. Returning alpha_lo.");
    }
    return alpha_lo;
}

bool StrongWolfe::isInvalid(double alpha_lo, double alpha_hi, double alpha) {
    bool isNotInBracket =
        !((alpha < alpha_hi && alpha > alpha_lo) || (alpha > alpha_hi && alpha < alpha_lo));
    bool isNotNumeric = !std::isfinite(alpha);
    double tol = 1e-8 * std::abs(alpha_hi - alpha_lo);
    bool tooClose = (std::abs(alpha - alpha_lo) < tol) || (std::abs(alpha - alpha_hi) < tol);
    return isNotInBracket || isNotNumeric || tooClose;
}
