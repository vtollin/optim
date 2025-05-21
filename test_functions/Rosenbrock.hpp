#include "optimization/ObjectiveFunction.hpp"
#include <Eigen/Dense>
#include <cmath>

class Rosenbrock : public ObjectiveFunction<2> {
  protected:
    double evaluateImpl(const Eigen::VectorXd &x) const override {
        double a = 1.0;
        double b = 100.0;
        return std::pow(a - x(0), 2) + b * std::pow(x(1) - x(0) * x(0), 2);
    }

    Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const override {
        double a = 1.0;
        double b = 100.0;
        Eigen::VectorXd grad(2);
        grad(0) = -2 * (a - x(0)) - 4 * b * x(0) * (x(1) - x(0) * x(0));
        grad(1) = 2 * b * (x(1) - x(0) * x(0));
        return grad;
    }
};
