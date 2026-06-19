#pragma once
#include "optim/Functions.hpp"
#include <Eigen/Dense>

class NonConvex1D : public optim::DifferentiableFunction {
  public:
    NonConvex1D() : optim::DifferentiableFunction(1) {}

  protected:
    double evaluateImpl(const Eigen::VectorXd &x) const override {
        double xx = x(0);
        return xx * xx * xx * xx - xx * xx;
    }
    Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const override {
        double xx = x(0);
        Eigen::VectorXd grad(1);
        grad << 4 * xx * xx * xx - 2 * xx;
        return grad;
    }
};