#pragma once
#include "optimization/ObjectiveFunction.hpp"
#include <Eigen/Dense>

// start at (0, 0)
class Himmelblau : public ObjectiveFunction<2> {
  protected:
    double evaluateImpl(const Eigen::VectorXd &x) const override {
        double x0 = x(0);
        double x1 = x(1);
        return std::pow(x0 * x0 + x1 - 11, 2) + std::pow(x0 + x1 * x1 - 7, 2);
    }

    Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const override {
        double x0 = x(0);
        double x1 = x(1);
        double a = (x0 * x0 + x1 - 11);
        double b = (x0 + x1 * x1 - 7);
        Eigen::VectorXd grad(2);
        grad << 4 * a * x0 + 2 * b, 2 * a + 4 * b * x1;
        return grad;
    }
};