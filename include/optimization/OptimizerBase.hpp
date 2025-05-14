#pragma once
#include "ObjectiveFunction.hpp"
#include "OptimizationResult.hpp"
#include <Eigen/Dense>

class OptimizerBase {
public: 
    virtual OptimizationResult optimize(
        ObjectiveFunction& f, 
        const Eigen::VectorXd& x0) = 0; 

    virtual ~OptimizerBase() = default; 
};