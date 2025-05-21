#include "optimization/ObjectiveFunction.hpp"
#include <Eigen/Dense>

class PoorlyScaledQuadratic : public ObjectiveFunction<2> {
  protected:
    double evaluateImpl(const Eigen::VectorXd &x) const override {
        Eigen::Matrix2d diag;
        diag << 100.0, 0.0, 0.0, 1.0;
        return 0.5 * x.dot(diag * x);
    }

    Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const override {
        Eigen::Matrix2d diag;
        diag << 100.0, 0.0, 0.0, 1.0;
        return diag * x;
    }
};