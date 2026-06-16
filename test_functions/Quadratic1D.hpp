#pragma once
#include "optim/Functions.hpp"
#include <Eigen/Dense>

class Quadratic1D : public optim::DifferentiableFunction<1> {
  protected:
    double evaluateImpl(const Eigen::VectorXd &x) const override { return x(0) * x(0); }
    Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const override { return 2 * x; }
};