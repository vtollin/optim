#pragma once
#include "optimization/LineSearch/SearchStrategyBase.hpp"
#include "optimization/OptimizationUtils.hpp"
#include <Eigen/Dense>

class ObjectiveFunctionBase;

namespace LineSearch {
struct ArmijoConfig {
    enum class TrialStepOpts {
        GEOMETRIC,
        GUARDED_INTERPOLATION
    } strategy = ArmijoConfig::TrialStepOpts::GUARDED_INTERPOLATION;
    double alpha_init = 1.0;
    double rho = 0.5;
    int max_iters = 20;
    double c = 1e-4;
};

class ArmijoBacktracking : public SearchStrategyBase {
  public:
    explicit ArmijoBacktracking(const ArmijoConfig &config);

    Eigen::VectorXd computeStep(const ObjectiveFunctionBase &f, const Eigen::VectorXd &x,
                                const Eigen::VectorXd &direction, const Eigen::VectorXd &gradient,
                                double alpha_override = -1.0) override;

  private:
    ArmijoConfig config_;
    double nextTrialStep(double alpha, double phi, double phi0, double phi_prime0,
                         double alpha_prev, double phi_prev, double alpha_min);
    double quadraticInterpolation(double endpoint_lo, double endpoint_hi, double deriv,
                                  double alpha);
    double cubicInterpolation(double endpoint_lo, double endpoint_hi, double midpoint, double deriv,
                              double alpha_prev, double alpha);
};
} // namespace LineSearch