#pragma once
#include <Eigen/Dense>

class ObjectiveFunction;

class LineSearch {
  public:
    virtual double chooseStep(ObjectiveFunction &f, const Eigen::VectorXd &x,
                              const Eigen::VectorXd &direction, const Eigen::VectorXd &gradient,
                              double alpha_override = -1.0) = 0;

    virtual ~LineSearch() = default;
};