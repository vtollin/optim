#pragma once
#include "optim/OptimizationResult.hpp"
#include "optim/trustregion/TrustRegionBase.hpp"
#include <Eigen/Dense>

namespace optim::logger {
class Logger;
}

namespace optim { class TwiceDifferentiableFunction; }

namespace optim::trustregion {
class Dogleg : public TrustRegionBase {
  public:
    Dogleg(int max_iterations = 100, optim::ConvergenceCriteria criteria = {},
           double delta_init = -1.0, double delta_max = 1000.0, double eta = 0.10,
           optim::logger::Logger *logger = nullptr);

    optim::OptimizationResult optimize(const optim::TwiceDifferentiableFunction &f,
                                       const Eigen::VectorXd &x0) override;
};
} // namespace optim::trustregion
