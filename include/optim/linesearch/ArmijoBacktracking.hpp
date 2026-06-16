#pragma once
#include "optim/linesearch/SearchStrategyBase.hpp"
#include "optim/OptimizerUtility.hpp"
#include <Eigen/Dense>
#include <optional>

namespace optim::abstract { class DifferentiableFunction; }

namespace optim::linesearch {
struct ArmijoConfig {
    double alpha_init = 1.0;
    int max_iters = 20;
    double c1 = 1e-4;
};

class ArmijoBacktracking : public SearchStrategyBase {
  public:
    explicit ArmijoBacktracking(const ArmijoConfig &config,
                                std::shared_ptr<optim::logger::Logger> logger = nullptr);

    double computeStep(const optim::abstract::DifferentiableFunction &f, const Eigen::VectorXd &x,
                       const Eigen::VectorXd &direction, const Eigen::VectorXd &gradient) override;
    void setConfig(const ArmijoConfig &cfg);

  private:
    ArmijoConfig config_;
    double nextTrialStep(double alpha, double phi, std::optional<double> alpha_prev,
                         std::optional<double> phi_prev, double phi0, double phi_prime0,
                         double alpha_min);
};
} // namespace optim::linesearch
