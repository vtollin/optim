#pragma once
#include <Eigen/Dense>

class ObjectiveFunctionBase {
  public:
    virtual ~ObjectiveFunctionBase() = default;

    virtual double evaluate(const Eigen::VectorXd &x) const = 0;
    virtual Eigen::VectorXd gradient(const Eigen::VectorXd &x) const = 0;
    virtual int sourceDimension() const = 0;
};