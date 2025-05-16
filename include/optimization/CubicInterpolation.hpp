#pragma once
#include "LineSearch.hpp"

class CubicInterpolation : public LineSearch {
  public:
    CubicInterpolation();

    double chooseStep(ObjectiveFunction &f, const Eigen::VectorXd &x,
                      const Eigen::VectorXd &direction, const Eigen::VectorXd &gradient,
                      double alpha_override = 1.0) override;
};