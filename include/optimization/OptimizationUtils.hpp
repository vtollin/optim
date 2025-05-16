#pragma once
#include "ObjectiveFunction.hpp"
#include <Eigen/Dense>

inline double directionalDerivative(ObjectiveFunction &f, const Eigen::VectorXd &x,
                                    const Eigen::VectorXd &direction) {
    return f.gradient(x).dot(direction);
}