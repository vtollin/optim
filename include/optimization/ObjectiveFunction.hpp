#pragma once 
#include <Eigen/Dense>

class ObjectiveFunction {
public:
    virtual double evaluate(const Eigen::VectorXd& x) = 0;
    virtual Eigen::VectorXd gradient(const Eigen::VectorXd& x) = 0;
    virtual ~ObjectiveFunction() = default; 
};