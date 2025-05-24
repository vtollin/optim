#pragma once
#include "optimization/ObjectiveFunction.hpp"
#include <Eigen/Dense>

class Quad10 : public ObjectiveFunction<1> {
  protected:
    double evaluateImpl(const Eigen::VectorXd &x) const override { return 10 * x(0) * x(0); }
    Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const override {
        Eigen::VectorXd grad(1);
        grad << 20 * x(0);
        return grad;
    }
};