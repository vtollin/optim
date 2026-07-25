#pragma once
#include "optim/OptimizationResult.hpp"
#include "optim/trustregion/TrustRegionBase.hpp"
#include <Eigen/Dense>
#include <memory>

namespace optim {
class TwiceDifferentiableFunction;
}

namespace optim::trustregion {
class MoreSorensen : public TrustRegionBase {
  public:
    MoreSorensen(int max_iteration = 100, optim::ConvergenceCriteria criteria = {},
                 TrustRegionConfig config = {});
    ~MoreSorensen();

  protected:
    SubproblemResult solveSubproblem(const Eigen::VectorXd &grad, const Eigen::MatrixXd &B,
                                     double delta) override;
};
} // namespace optim::trustregion
