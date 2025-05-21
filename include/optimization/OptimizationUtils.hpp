#pragma once
#include "ObjectiveFunction.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <limits>

namespace Utility {
inline double directionalDerivative(const ObjectiveFunctionBase &f, const Eigen::VectorXd &x,
                                    const Eigen::VectorXd &direction) {
    return f.gradient(x).dot(direction);
}

constexpr double epsilon = std::numeric_limits<double>::epsilon();
constexpr double sqrt_epsilon = 1.4901161193847656e-08;
} // namespace Utility