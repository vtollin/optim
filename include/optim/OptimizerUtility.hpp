#pragma once
#include "AbstractFunctions.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <limits>

namespace optim::abstract { class DifferentiableFunction; }

namespace optim::utility {
inline double directionalDerivative(const optim::abstract::DifferentiableFunction &f,
                                    const Eigen::VectorXd &x,
                                    const Eigen::VectorXd &direction) {
    return f.gradient(x).dot(direction);
}

constexpr double epsilon = std::numeric_limits<double>::epsilon();
inline const double sqrt_epsilon = std::sqrt(epsilon);
} // namespace optim::utility
