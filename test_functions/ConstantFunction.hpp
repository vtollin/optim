#pragma once
#include "optimization/ObjectiveFunction.hpp"
#include <Eigen/Dense>

class ConstantFunction : public ObjectiveFunction<2> {
  protected:
    double evaluateImpl(const Eigen::VectorXd &x) const override { return 1; }

    Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const override {
        return Eigen::VectorXd::Zero(2);
    }
};