#pragma once
#include "optimization/ObjectiveFunction.hpp"
#include <Eigen/Dense>

// start at (1,1)
class Beale : public ObjectiveFunction<2> {
  protected:
    double evaluateImpl(const Eigen::VectorXd &x) const override {
        double x0 = x(0), x1 = x(1);
        double t1 = 1.5 - x0 + x0 * x1;
        double t2 = 2.25 - x0 + x0 * x1 * x1;
        double t3 = 2.625 - x0 + x0 * x1 * x1 * x1;
        return t1 * t1 + t2 * t2 + t3 * t3;
    }

    Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const override {
        double x0 = x(0), x1 = x(1);
        double t1 = 1.5 - x0 + x0 * x1;
        double t2 = 2.25 - x0 + x0 * x1 * x1;
        double t3 = 2.625 - x0 + x0 * x1 * x1 * x1;

        Eigen::VectorXd grad(2);
        grad(0) = 2 * t1 * (-1 + x1) + 2 * t2 * (-1 + x1 * x1) + 2 * t3 * (-1 + x1 * x1 * x1);
        grad(1) = 2 * t1 * (x0) + 4 * t2 * (x0 * x1) + 6 * t3 * (x0 * x1 * x1);
        return grad;
    }
};
