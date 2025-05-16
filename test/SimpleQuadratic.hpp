#pragma once
#include "optimization/ObjectiveFunction.hpp"
#include <Eigen/Dense>

class SimpleQuadratic : public ObjectiveFunction {
  public:
    double evaluate(const Eigen::VectorXd &x) override { return 0.5 * x.squaredNorm(); }

    Eigen::VectorXd gradient(const Eigen::VectorXd &x) override { return x; }
};