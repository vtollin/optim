#pragma once
#include "BMatrixHandler.hpp"

namespace optim::trustregion {
class ExactHessianHandler : public BMatrixHandler {
  public:
    ExactHessianHandler();
    Eigen::MatrixXd initialize(const optim::abstract::DifferentiableFunction &f,
                               const Eigen::VectorXd &x) override;
    Eigen::MatrixXd getB(const optim::abstract::DifferentiableFunction &f,
                         const Eigen::VectorXd &x) override;
};
} // namespace optim::trustregion
