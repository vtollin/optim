#pragma once
#include "optim/AbstractFunctions.hpp"
#include <Eigen/Dense>

namespace optim::trustregion {
class BMatrixHandler {
  public:
    virtual ~BMatrixHandler() = default;
    virtual Eigen::MatrixXd getB(const optim::abstract::DifferentiableFunction &f,
                                 const Eigen::VectorXd &x) = 0;
    virtual Eigen::MatrixXd initialize(const optim::abstract::DifferentiableFunction &f,
                                       const Eigen::VectorXd &x) = 0;
};
} // namespace optim::trustregion
