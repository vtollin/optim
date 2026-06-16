#pragma once
#include "optim/OptimizerBase.hpp"
#include "optim/OptimizationResult.hpp"
#include "optim/trustregion/QuadraticModel.hpp"
#include <Eigen/Dense>
#include <stdexcept>

namespace optim::logger {
class Logger;
}

namespace optim::abstract { class TwiceDifferentiableFunction; }

namespace optim::trustregion {

struct UpdateResult {
    double rho;
    double delta_old;
    bool accepted;
};

class TrustRegionBase : public optim::OptimizerBase {
  public:
    virtual optim::OptimizationResult optimize(const optim::abstract::TwiceDifferentiableFunction &f,
                                               const Eigen::VectorXd &x0) = 0;

  protected:
    double delta_;
    double delta_max_;
    double eta_;

    TrustRegionBase(int max_iterations, optim::ConvergenceCriteria criteria, double delta_init,
                    double delta_max, double eta, std::shared_ptr<optim::logger::Logger> logger)
        : delta_(delta_init)
        , delta_max_(delta_max)
        , eta_(eta)
        , OptimizerBase(max_iterations, criteria, logger) {
        if (delta_max_ <= 0.0) {
            throw std::invalid_argument("[TrustRegionBase]: delta_max must be positive");
        }
        if (eta_ <= 0.0) {
            throw std::invalid_argument("[TrustRegionBase]: eta must be positive");
        }
        if (delta_ == 0.0) {
            throw std::invalid_argument("[TrustRegionBase]: delta_init cannot be zero");
        }
    };

    UpdateResult update(const optim::abstract::TwiceDifferentiableFunction &f,
                        const QuadraticModel &m,
                        const Eigen::VectorXd &x, const Eigen::VectorXd &step);
};
} // namespace optim::trustregion
