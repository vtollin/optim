#pragma once
#include <Eigen/Dense>
#include "ObjectiveFunction.hpp"

class LineSearch {
    virtual double chooseStep(
        ObjectiveFunction& f, 
        const Eigen::VectorXd& x,
        const Eigen::VectorXd& direction
    ) = 0;

    virtual ~LineSearch() = default; 
};