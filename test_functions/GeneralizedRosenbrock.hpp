#pragma once
#include "optim/Functions.hpp"
#include <Eigen/Dense>
#include <cmath>

class GeneralizedRosenbrock : public optim::TwiceDifferentiableFunction {
  public:
    GeneralizedRosenbrock(int n) : TwiceDifferentiableFunction(n) {}

  protected:
    double evaluateImpl(const Eigen::VectorXd &x) const override {
        double a = 1.0;
        double b = 100.0;
        int n = sourceDimension();
        double sum = 0.0;
        for (int i = 0; i < n - 1; ++i) {
            sum += b * std::pow(x(i + 1) - x(i) * x(i), 2) + std::pow(a - x(i), 2);
        }
        return sum;
    }

    Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const override {
        double a = 1.0;
        double b = 100.0;
        int n = sourceDimension();
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(n);
        for (int i = 0; i < n - 1; ++i) {
            grad(i) += -4.0 * b * x(i) * (x(i + 1) - x(i) * x(i)) - 2.0 * (a - x(i));
            grad(i + 1) += 2.0 * b * (x(i + 1) - x(i) * x(i));
        }
        return grad;
    }

    Eigen::MatrixXd hessianImpl(const Eigen::VectorXd &x) const override {
        double b = 100.0;
        int n = sourceDimension();
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
        for (int i = 0; i < n - 1; ++i) {
            H(i, i) += -4.0 * b * x(i + 1) + 12.0 * b * x(i) * x(i) + 2.0;
            H(i + 1, i + 1) += 2.0 * b;
            H(i, i + 1) += -4.0 * b * x(i);
            H(i + 1, i) += -4.0 * b * x(i);
        }
        return H;
    }
};