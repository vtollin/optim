#pragma once
#include "optim/OptimizationResult.hpp"
#include "optim/trustregion/TrustRegionBase.hpp"
#include <Eigen/Dense>
#include <memory>

namespace optim {
class TwiceDifferentiableFunction;
}

namespace optim::trustregion {
class BMatrixHandler;
}

namespace optim::trustregion {
enum BMatrixConfig { EXACT, APPROXIMATE };
class MoreSorensen : public TrustRegionBase {
  public:
    MoreSorensen(int max_iterations, BMatrixConfig cfg, optim::ConvergenceCriteria criteria = {},
                 TrustRegionConfig config = {});
    ~MoreSorensen();

  protected:
    SubproblemResult solveSubproblem(const Eigen::VectorXd &grad, const Eigen::MatrixXd &B,
                                     double delta) override;
    Eigen::MatrixXd initializeB(const optim::TwiceDifferentiableFunction &f,
                                const Eigen::VectorXd &x0) override;
    Eigen::MatrixXd updateB(const optim::TwiceDifferentiableFunction &f,
                            const Eigen::VectorXd &x) override;

  private:
    std::unique_ptr<BMatrixHandler> b_handler_;
};
} // namespace optim::trustregion
