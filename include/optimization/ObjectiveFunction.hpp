#pragma once
#include <Eigen/Dense>

class ObjectiveFunction {
  public:
    virtual double evaluate(const Eigen::VectorXd &x) = 0;
    virtual Eigen::VectorXd gradient(const Eigen::VectorXd &x) = 0;

    virtual Eigen::MatrixXd hessian(const Eigen::VectorXd &x) {
        throw std::runtime_error("Hessian not implemented for this function.");
    }

    virtual bool supportsHessian() const { return false; }

    virtual ~ObjectiveFunction() = default;
};