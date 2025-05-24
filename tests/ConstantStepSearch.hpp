#pragma once
#include "optimization/LineSearch/SearchStrategyBase.hpp"
#include <Eigen/Dense>

class ConstantStepSearch : public LineSearch::SearchStrategyBase {
  public:
    ConstantStepSearch(double alpha) : alpha_(alpha) {}
    Eigen::VectorXd computeStep(const ObjectiveFunctionBase &f, const Eigen::VectorXd &x,
                                const Eigen::VectorXd &direction,
                                const Eigen::VectorXd &grad) override {
        return alpha_ * direction;
    }

  private:
    double alpha_;
};
