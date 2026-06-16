#pragma once
#include "optim/Functions.hpp"
#include <Eigen/Dense>

class ConstantFunction : public optim::TwiceDifferentiableFunction<2> {
  protected:
    double evaluateImpl(const Eigen::VectorXd &x) const override { return 1; }

    Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const override {
        return Eigen::VectorXd::Zero(2);
    }

    Eigen::MatrixXd hessianImpl(const Eigen::VectorXd &x) const override {
        return Eigen::MatrixXd::Zero(2, 2);
    }
};