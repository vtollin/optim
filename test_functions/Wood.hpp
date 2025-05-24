#pragma once
#include "optimization/ObjectiveFunction.hpp"
#include <Eigen/Dense>

// start at (-3,-1,-3,-1)
class Wood : public ObjectiveFunction<4> {
  protected:
    double evaluateImpl(const Eigen::VectorXd &x) const override {
        double x1 = x(0), x2 = x(1), x3 = x(2), x4 = x(3);
        double t1 = x2 - x1 * x1;
        double t2 = x4 - x3 * x3;
        double term1 = 100 * std::pow(t1, 2) + std::pow(1 - x1, 2);
        double term2 = 90 * std::pow(t2, 2) + std::pow(1 - x3, 2);
        double term3 =
            10.1 * (std::pow(x2 - 1, 2) + std::pow(x4 - 1, 2)) + 19.8 * (x2 - 1) * (x4 - 1);
        return term1 + term2 + term3;
    }

    Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const override {
        double x1 = x(0), x2 = x(1), x3 = x(2), x4 = x(3);
        double t1 = x2 - x1 * x1;
        double t2 = x4 - x3 * x3;

        Eigen::VectorXd grad(4);
        grad(0) = -400 * t1 * x1 - 2 * (1 - x1);
        grad(1) = 200 * t1 + 20.2 * (x2 - 1) + 19.8 * (x4 - 1);
        grad(2) = -360 * t2 * x3 - 2 * (1 - x3);
        grad(3) = 180 * t2 + 20.2 * (x4 - 1) + 19.8 * (x2 - 1);
        return grad;
    }

    Eigen::MatrixXd hessianImpl(const Eigen::VectorXd &x) const override {
        // unpack
        const double x1 = x(0), x2 = x(1), x3 = x(2), x4 = x(3);
        // common expressions
        double t1 = x2 - x1 * x1;
        double t2 = x4 - x3 * x3;

        Eigen::MatrixXd H(4, 4);
        H.setZero();

        H(0, 0) = 1200.0 * x1 * x1 - 400.0 * x2 + 2.0;
        H(0, 1) = -400.0 * x1;
        H(1, 0) = H(0, 1);

        H(1, 1) = 200.0 + 20.2;
        H(1, 3) = 19.8;
        H(3, 1) = H(1, 3);

        H(2, 2) = 1080.0 * x3 * x3 - 360.0 * x4 + 2.0;
        H(2, 3) = -360.0 * x3;
        H(3, 2) = H(2, 3);

        H(3, 3) = 180.0 + 20.2;

        return H;
    }
};
