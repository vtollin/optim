#pragma once
#include <Eigen/Dense>

class ObjectiveFunctionBase;

namespace LineSearch {
class SearchStrategyBase {
  public:
    virtual ~SearchStrategyBase() = default;
    virtual Eigen::VectorXd computeStep(const ObjectiveFunctionBase &f, const Eigen::VectorXd &x,
                                        const Eigen::VectorXd &direction,
                                        const Eigen::VectorXd &gradient,
                                        double alpha_override = -1.0) = 0;
};
}