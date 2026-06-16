#pragma once
#include <Eigen/Dense>

namespace optim::abstract {

class Function {
public:
    virtual ~Function() = default;
    virtual double evaluate(const Eigen::VectorXd &x) const = 0;
    virtual int sourceDimension() const = 0;
};

class DifferentiableFunction : public Function {
public:
    virtual Eigen::VectorXd gradient(const Eigen::VectorXd &x) const = 0;
};

class TwiceDifferentiableFunction : public DifferentiableFunction {
public:
    virtual Eigen::MatrixXd hessian(const Eigen::VectorXd &x) const = 0;
};

} // namespace optim::abstract
