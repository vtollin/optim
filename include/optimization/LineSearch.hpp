#pragma once
#include <Eigen/Dense>
#include "ObjectiveFunction.hpp"

// read step length selection algorithms 
class LineSearch {
public:
    virtual double chooseStep(
        ObjectiveFunction& f, 
        const Eigen::VectorXd& x,
        const Eigen::VectorXd& direction,
        const Eigen::VectorXd& gradient
    ) = 0;

    virtual ~LineSearch() = default; 
};