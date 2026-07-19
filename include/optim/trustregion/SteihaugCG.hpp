#pragma once
#include "optim/OptimizationResult.hpp"
#include "optim/trustregion/TrustRegionBase.hpp"
#include <Eigen/Dense>

namespace optim::trustregion {
class SteihaugCG : public TrustRegionBase {
  public:
    SteihaugCG(int max_iterations = 1000, optim::ConvergenceCriteria criteria = {},
               TrustRegionConfig config = {});

  private:
    SubproblemResult solveSubproblem(const Eigen::VectorXd &grad, const Eigen::MatrixXd &B,
                                     double delta) override;
    double computeBoundaryTau(const Eigen::VectorXd &z, const Eigen::VectorXd &d, double delta);
};
} // namespace optim::trustregion