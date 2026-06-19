#pragma once
#include "optim/Functions.hpp"
#include <Eigen/Dense>

class IllCondQuad : public optim::TwiceDifferentiableFunction {
  public:
    IllCondQuad() : optim::TwiceDifferentiableFunction(2) {}

  protected:
    double evaluateImpl(const Eigen::VectorXd &x) const override {
        Eigen::Matrix2d diag;
        diag << 1000.0, 0.0, 0.0, 1.0;
        return 0.5 * x.dot(diag * x);
    }

    Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const override {
        Eigen::Matrix2d diag;
        diag << 1000.0, 0.0, 0.0, 1.0;
        return diag * x;
    }

    Eigen::MatrixXd hessianImpl(const Eigen::VectorXd &x) const override {
        Eigen::Matrix2d D;
        D << 1000.0, 0.0, 0.0, 1.0;
        return D;
    }
};