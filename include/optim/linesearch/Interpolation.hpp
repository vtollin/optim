#pragma once
#include <cmath>
#include <limits>

namespace optim::linesearch::interpolation {

// Fits a quadratic through (alpha_lo, phi_lo) with derivative phi_prime_lo at alpha_lo,
// and through (alpha_hi, phi_hi). Returns the minimizer.
// For Armijo backtracking call with alpha_lo = 0.0; for Wolfe zoom use general alpha_lo.
inline double quadraticMinimizer(double alpha_lo, double alpha_hi, double phi_lo, double phi_hi,
                                 double phi_prime_lo) {
    double d = alpha_hi - alpha_lo;
    return alpha_lo - phi_prime_lo * d * d / (2.0 * (phi_hi - phi_lo - phi_prime_lo * d));
}

// Fits a cubic through (0, phi0) with derivative phi_prime0 at 0, and through
// (alpha_prev, phi_prev) and (alpha, phi). Returns the minimizer.
// Used in Armijo backtracking when a previous step is available.
inline double cubicMinimizer(double phi0, double phi_prev, double phi, double phi_prime0,
                             double alpha_prev, double alpha) {
    double a_num = alpha_prev * alpha_prev * (phi - phi0 - phi_prime0 * alpha) -
                   alpha * alpha * (phi_prev - phi0 - phi_prime0 * alpha_prev);
    double b_num = -alpha_prev * alpha_prev * alpha_prev * (phi - phi0 - phi_prime0 * alpha) +
                   alpha * alpha * alpha * (phi_prev - phi0 - phi_prime0 * alpha_prev);
    double denom = alpha_prev * alpha_prev * alpha * alpha * (alpha - alpha_prev);
    double a = a_num / denom;
    double b = b_num / denom;
    // phi0, phi, and phi_prev lie on a quadratic. Cubic formula degenerates (a = 0), but quadratic
    // has minimizer at -(phi_prime0 / 2b) since b > 0.
    if (std::abs(a) < std::sqrt(std::numeric_limits<double>::epsilon()) * std::abs(b) && b > 0) {
        return -phi_prime0 / (2 * b);
    }
    return (-b + std::sqrt(b * b - 3.0 * a * phi_prime0)) / (3.0 * a);
}

// Fits a Hermite cubic through (alpha_lo, phi_lo) and (alpha_hi, phi_hi) with derivatives
// phi_prime_lo and phi_prime_hi at the respective endpoints. Returns the minimizer.
// N&W eq. 3.59. Used in the Strong Wolfe zoom phase.
inline double cubicHermiteMinimizer(double alpha_lo, double alpha_hi, double phi_lo, double phi_hi,
                                    double phi_prime_lo, double phi_prime_hi) {
    double d1 = phi_prime_lo + phi_prime_hi - 3.0 * (phi_lo - phi_hi) / (alpha_lo - alpha_hi);
    double d2 = std::sqrt(d1 * d1 - phi_prime_lo * phi_prime_hi);
    double num = phi_prime_hi + ((alpha_hi > alpha_lo) ? +d2 : -d2) - d1;
    double denom = phi_prime_hi - phi_prime_lo + 2.0 * ((alpha_hi > alpha_lo) ? +d2 : -d2);
    return alpha_hi - (alpha_hi - alpha_lo) * num / denom;
}

} // namespace optim::linesearch::interpolation
