#include "optimization/LineSearch/StrongWolfe.hpp"
#include "optimization/ObjectiveFunctionBase.hpp"
#include "optimization/OptimizationUtils.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <limits>

using namespace LineSearch;

StrongWolfe::StrongWolfe(const WolfeConfig &config) : config_(config) {
    if (config.alpha_init <= 0.0) {
        throw std::invalid_argument("StrongWolfe config: alpha_init must be > 0.");
    }
    if (config.alpha_max <= 0.0) {
        throw std::invalid_argument("StrongWolfe config: alpha_max must be > 0.");
    }
    if (config.alpha_init > config.alpha_max) {
        throw std::invalid_argument("StrongWolfe config: alpha_init must not exceed alpha_max.");
    }
    if (config.max_iters <= 0) {
        throw std::invalid_argument("StrongWolfe config: max_iters must be positive.");
    }
    if (config.rho <= 1.0) {
        throw std::invalid_argument("StrongWolfe config: rho must be > 1.0 for forward step.");
    }
    if (config.c1 <= 0.0 || config.c1 >= 1.0) {
        throw std::invalid_argument("StrongWolfe config: c1 must be in (0, 1).");
    }
    if (config.c2 <= config.c1 || config.c2 >= 1.0) {
        throw std::invalid_argument("StrongWolfe config: c2 must be in (c1, 1).");
    }
}

Eigen::VectorXd StrongWolfe::computeStep(const ObjectiveFunctionBase &f, const Eigen::VectorXd &x0,
                                         const Eigen::VectorXd &direction,
                                         const Eigen::VectorXd &gradient, double alpha_override) {
    double phi0 = f.evaluate(x0);
    double phi_prime0 = Utility::directionalDerivative(f, x0, direction);
    double alpha = (alpha_override > 0.0) ? alpha_override : config_.alpha_init;
    double alpha_prev = 0.0;
    double alpha_next;
    int stall_counter = 0;
    // Initialize phi_prev_ to max to skip phi > phi_prev_ on first iteration
    double phi_prev = std::numeric_limits<double>::max();
    double phi = f.evaluate(x0 + alpha * direction);
    double best_alpha = alpha;
    double best_phi = phi;
    for (int k = 0; k < config_.max_iters; ++k) {
        if (phi > phi0 + config_.c1 * alpha * phi_prime0 || phi > phi_prev) {
            return zoom(alpha_prev, alpha, f, x0, direction, phi0, phi_prime0) * direction;
        }
        double phi_prime = Utility::directionalDerivative(f, x0 + alpha * direction, direction);
        if (std::fabs(phi_prime) <= -config_.c2 * phi_prime0) {
            return alpha * direction;
        }
        if (phi_prime >= 0) {
            return zoom(alpha, alpha_prev, f, x0, direction, phi0, phi_prime0) * direction;
        }
        alpha_next = alpha * config_.rho; // compute next step
        if (alpha_next > config_.alpha_max) {
            if (config_.isVerbose) {
                std::cerr << "[StrongWolfe] Warning: Alpha exceeded alpha_max: returning best step "
                             "found.\n";
            }
            break;
        }
        alpha_prev = alpha;
        phi_prev = phi;
        phi = f.evaluate(x0 + alpha_next * direction);
        if (phi < best_phi) {
            best_alpha = alpha_next;
            best_phi = phi;
        }
        double tol = Utility::epsilon * (std::max(std::abs(phi), std::abs(phi_prev)) + 1.0);
        if (std::abs(phi - phi_prev) < tol) {
            ++stall_counter;
            if (stall_counter == 10) {
                if (config_.isVerbose) {
                    std::cerr
                        << "[StrongWolfe] Warning: Change in phi fell below machine precision "
                           "for 10 iterations. Returning best step found.\n";
                }
            }
            break;
        } else {
            stall_counter = 0;
        }
        alpha = alpha_next;
    }
    return best_alpha * direction;
}

double StrongWolfe::zoom(double alpha_lo, double alpha_hi, const ObjectiveFunctionBase &f,
                         const Eigen::VectorXd &x0, const Eigen::VectorXd &direction, double phi0,
                         double phi_prime0) {
    for (int i = 0; i < 15; ++i) { // add max iterations or one of the stopping conditions
        double phi_lo = f.evaluate(x0 + alpha_lo * direction);
        double phi_hi = f.evaluate(x0 + alpha_hi * direction);
        double phi_prime_lo =
            Utility::directionalDerivative(f, x0 + alpha_lo * direction, direction);
        double phi_prime_hi =
            Utility::directionalDerivative(f, x0 + alpha_hi * direction, direction);
        double alpha =
            cubicInterpolationZoom(alpha_lo, alpha_hi, phi_lo, phi_hi, phi_prime_lo, phi_prime_hi);
        if (isInvalid(alpha_lo, alpha_hi, alpha)) { // fall back to quadratic
            alpha = quadraticInterpolationZoom(alpha_lo, alpha_hi, phi_lo, phi_hi, phi_prime_lo);
        }
        if (isInvalid(alpha_lo, alpha_hi, alpha)) { // fall back to bisection
            alpha = 0.5 * (alpha_hi + alpha_lo);
        }
        double phi = f.evaluate(x0 + alpha * direction);
        if (phi > phi0 + config_.c1 * alpha * phi_prime0 || phi >= phi_lo) {
            alpha_hi = alpha;
        } else {
            double phi_prime = Utility::directionalDerivative(f, x0 + alpha * direction, direction);
            if (std::abs(phi_prime) <= -config_.c2 * phi_prime0) {
                return alpha;
            }
            if (phi_prime * (alpha_hi - alpha_lo) >= 0) {
                alpha_hi = alpha_lo;
            }
            alpha_lo = alpha;
        }
        double tol = Utility::sqrt_epsilon * (std::max(alpha_lo, alpha_hi) + 1.0);
        if (std::abs(alpha_hi - alpha_lo) < tol) { // break if bracket falls below tolerance
            break;
        }
    }
    if (config_.isVerbose) {
        std::cerr << "[StrongWolfe] Warning: zoom() failed to converge. Returning alpha_lo.\n";
    }
    return alpha_lo;
}

double StrongWolfe::cubicInterpolationZoom(double alpha_lo, double alpha_hi, double phi_lo,
                                           double phi_hi, double phi_prime_lo,
                                           double phi_prime_hi) {
    // return minimizer
    double d1 = phi_prime_lo + phi_prime_hi - 3 * (phi_lo - phi_hi) / (alpha_lo - alpha_hi);
    double d2 = std::sqrt(d1 * d1 - phi_prime_lo * phi_prime_hi);
    double numerator = phi_prime_hi + ((alpha_hi > alpha_lo) ? +d2 : -d2) - d1;
    double denominator = phi_prime_hi - phi_prime_lo + 2 * ((alpha_hi > alpha_lo) ? +d2 : -d2);
    double alpha = alpha_hi - (alpha_hi - alpha_lo) * numerator / denominator;
    return alpha;
}

double StrongWolfe::quadraticInterpolationZoom(double alpha_lo, double alpha_hi, double phi_lo,
                                               double phi_hi, double phi_prime_lo) {
    // return minimizer
    double d = alpha_hi - alpha_lo;
    double num = phi_prime_lo * d * d;
    double denom = 2.0 * (phi_hi - phi_lo - phi_prime_lo * d);
    double alpha = alpha_lo - num / denom;
    return alpha;
}

bool StrongWolfe::isInvalid(double alpha_lo, double alpha_hi, double alpha) {
    bool isNotInBracket =
        !((alpha < alpha_hi && alpha > alpha_lo) || (alpha > alpha_hi && alpha < alpha_lo));
    bool isNotNumeric = !std::isfinite(alpha);
    double tol = 1e-8 * std::abs(alpha_hi - alpha_lo);
    bool tooClose = (std::abs(alpha - alpha_lo) < tol) || (std::abs(alpha - alpha_hi) < tol);
    return isNotInBracket || isNotNumeric || tooClose;
}
