#pragma once
#include "optimization/OptimizationResult.hpp"
#include <Eigen/Dense>

class ObjectiveFunctionBase;

class OptimizerBase {
  public:
    virtual ~OptimizerBase() = default;
    virtual OptimizationResult optimize(const ObjectiveFunctionBase &f,
                                        const Eigen::VectorXd &x0) = 0;
};