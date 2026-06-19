#pragma once
#include "optim/Functions.hpp"
#include <Eigen/Dense>

namespace optim::trustregion {
class BMatrixHandler {
  public:
    virtual ~BMatrixHandler() = default;
    virtual Eigen::MatrixXd getB(const optim::DifferentiableFunction &f,
                                 const Eigen::VectorXd &x) = 0;
    virtual Eigen::MatrixXd initialize(const optim::DifferentiableFunction &f,
                                       const Eigen::VectorXd &x) = 0;
};
} // namespace optim::trustregion
