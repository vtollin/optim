#pragma once
#include "optim/Functions.hpp"
#include <Eigen/Dense>

class ConvexQuadratic : public optim::TwiceDifferentiableFunction<2> {
  protected:
    double evaluateImpl(const Eigen::VectorXd &x) const override { return 0.5 * x.squaredNorm(); }

    Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const override { return x; }

    Eigen::MatrixXd hessianImpl(const Eigen::VectorXd &x) const override {
        return Eigen::Matrix2d::Identity();
    }
};