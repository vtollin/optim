#pragma once
#include "optim/OptimizationResult.hpp"
#include "optim/trustregion/TrustRegionBase.hpp"
#include <Eigen/Dense>

namespace optim {
class TwiceDifferentiableFunction;
}

namespace optim::trustregion {
class Dogleg : public TrustRegionBase {
  public:
    Dogleg(int max_iterations = 100, optim::ConvergenceCriteria criteria = {},
           double delta_init = -1.0, double delta_max = 1000.0, double eta = 0.10);

  protected:
    SubproblemResult solveSubproblem(const Eigen::VectorXd &grad, const Eigen::MatrixXd &B,
                                     double delta) override;
};
} // namespace optim::trustregion
