#pragma once
#include "optim/Functions.hpp"
#include <Eigen/Dense>

class Saddle : public optim::TwiceDifferentiableFunction {
  protected:
    double evaluateImpl(const Eigen::VectorXd &x) const override {
        return x(0) * x(0) - x(1) * x(1);
    };

    Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const override {
        Eigen::VectorXd grad(2);
        grad << 2.0 * x(0), -2.0 * x(1);
        return grad;
    }

    Eigen::MatrixXd hessianImpl(const Eigen::VectorXd &x) const override {
        Eigen::VectorXd diag(2);
        diag << 2.0, -2.0;
        return diag.asDiagonal();
    }
};