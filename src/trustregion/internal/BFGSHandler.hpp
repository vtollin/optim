#pragma once
#include "BMatrixHandler.hpp"

namespace optim::trustregion {
class BFGSHandler : public BMatrixHandler {
  public:
    BFGSHandler();
    Eigen::MatrixXd getB(const optim::abstract::DifferentiableFunction &f,
                         const Eigen::VectorXd &x) override;
    Eigen::MatrixXd initialize(const optim::abstract::DifferentiableFunction &f,
                               const Eigen::VectorXd &x) override;

  private:
    Eigen::VectorXd x_prev_;
    Eigen::VectorXd grad_prev_;
    Eigen::MatrixXd B_prev_;
};
} // namespace optim::trustregion
