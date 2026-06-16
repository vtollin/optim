#pragma once
#include "optim/Functions.hpp"
#include <Eigen/Dense>

// start at (1,1)
class Beale : public optim::TwiceDifferentiableFunction<2> {
  protected:
    double evaluateImpl(const Eigen::VectorXd &x) const override {
        double x0 = x(0), x1 = x(1);
        double t1 = 1.5 - x0 + x0 * x1;
        double t2 = 2.25 - x0 + x0 * x1 * x1;
        double t3 = 2.625 - x0 + x0 * x1 * x1 * x1;
        return t1 * t1 + t2 * t2 + t3 * t3;
    }

    Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const override {
        double x0 = x(0), x1 = x(1);
        double t1 = 1.5 - x0 + x0 * x1;
        double t2 = 2.25 - x0 + x0 * x1 * x1;
        double t3 = 2.625 - x0 + x0 * x1 * x1 * x1;

        Eigen::VectorXd grad(2);
        grad(0) = 2 * t1 * (-1 + x1) + 2 * t2 * (-1 + x1 * x1) + 2 * t3 * (-1 + x1 * x1 * x1);
        grad(1) = 2 * t1 * (x0) + 4 * t2 * (x0 * x1) + 6 * t3 * (x0 * x1 * x1);
        return grad;
    }

    Eigen::MatrixXd hessianImpl(const Eigen::VectorXd &x) const override {
        double x0 = x(0), x1 = x(1);
        double t1 = 1.5 - x0 + x0 * x1;
        double t2 = 2.25 - x0 + x0 * x1 * x1;
        double t3 = 2.625 - x0 + x0 * x1 * x1 * x1;

        double dt1_dx0 = x1 - 1, dt1_dx1 = x0;
        double dt2_dx0 = -1 + x1 * x1, dt2_dx1 = 2 * x0 * x1;
        double dt3_dx0 = -1 + x1 * x1 * x1, dt3_dx1 = 3 * x0 * x1 * x1;

        Eigen::Matrix2d H;
        H(0, 0) = 2 * (dt1_dx0 * dt1_dx0 + dt2_dx0 * dt2_dx0 + dt3_dx0 * dt3_dx0);
        H(0, 1) = 2 * (dt1_dx0 * dt1_dx1 + dt2_dx0 * dt2_dx1 + dt3_dx0 * dt3_dx1);
        H(1, 0) = H(0, 1);
        H(1, 1) = 2 * (dt1_dx1 * dt1_dx1 + dt2_dx1 * dt2_dx1 + dt3_dx1 * dt3_dx1);

        H(0, 1) += 2 * (t1 * 1 + t2 * (2 * x1) + t3 * (3 * x1 * x1));
        H(1, 0) = H(0, 1);
        H(1, 1) += 2 * (t2 * (2 * x0) + t3 * (6 * x0 * x1));

        return H;
    }
};
