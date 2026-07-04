#pragma once
#include "optim/Functions.hpp"
#include <Eigen/Dense>

class RankDeficientQuad : public optim::TwiceDifferentiableFunction {
  public:
    RankDeficientQuad() : TwiceDifferentiableFunction(2) {}

  protected:
    double evaluateImpl(const Eigen::VectorXd &x) const override { return x(0) * x(0); }
    Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const override {
        Eigen::VectorXd grad(2);
        grad << 2 * x(0), 0.0;
        return grad;
    }
    Eigen::MatrixXd hessianImpl(const Eigen::VectorXd &x) const override {
        Eigen::VectorXd diag(2);
        diag << 2.0, 0.0;
        return diag.asDiagonal();
    }
};