#pragma once
#include "optim/OptimizationResult.hpp"
#include "optim/trustregion/TrustRegionBase.hpp"
#include <Eigen/Dense>

namespace optim::logger {
class Logger;
}

namespace optim { class TwiceDifferentiableFunction; }

namespace optim::trustregion { class BMatrixHandler; }

namespace optim::trustregion {
enum BMatrixConfig { EXACT, APPROXIMATE };
class MoreSorensen : public TrustRegionBase {
  public:
    MoreSorensen(int max_iterations, optim::ConvergenceCriteria criteria, double delta_init,
                 double delta_max, double eta, BMatrixConfig cfg,
                 std::shared_ptr<optim::logger::Logger> logger = nullptr);
    ~MoreSorensen();

    optim::OptimizationResult optimize(const optim::TwiceDifferentiableFunction &f,
                                       const Eigen::VectorXd &x0) override;

  private:
    std::unique_ptr<BMatrixHandler> b_handler_;
    bool tryNewton(const Eigen::VectorXd &grad, const Eigen::MatrixXd &B, Eigen::VectorXd &p_out);
    Eigen::VectorXd newtonRootFind(const Eigen::MatrixXd &B, const Eigen::VectorXd &grad,
                                   double lambda1);
};
} // namespace optim::trustregion
