#pragma once
#include "optimization/LineSearch/SearchStrategyBase.hpp"
#include "optimization/OptimizationUtils.hpp"
#include <Eigen/Dense>

class ObjectiveFunctionBase;

namespace LineSearch {
struct WolfeConfig {
    double alpha_init = 1.0;
    double alpha_max = 10.0;
    double rho = 2.0;
    int max_iters = 20;
    double c1 = 1e-4;
    double c2 = 0.9;
    bool isVerbose = false;
};

class StrongWolfe : public SearchStrategyBase {
  public:
    StrongWolfe(const WolfeConfig &config);
    Eigen::VectorXd computeStep(const ObjectiveFunctionBase &f, const Eigen::VectorXd &x,
                                const Eigen::VectorXd &direction, const Eigen::VectorXd &gradient,
                                double alpha_override = -1.0) override;

  private:
    WolfeConfig config_;
    double zoom(double alpha_lo, double alpha_hi, const ObjectiveFunctionBase &f,
                const Eigen::VectorXd &x, const Eigen::VectorXd &direction, double phi0,
                double phi_prime0);
    double cubicInterpolationZoom(double alpha_lo, double alpha_hi, double phi_lo, double phi_hi,
                                  double phi_prime_lo, double phi_prime_hi);
    double quadraticInterpolationZoom(double alpha_lo, double alpha_hi, double phi_lo,
                                      double phi_hi, double phi_prime_lo);
    bool isInvalid(double alpha_lo, double alpha_hi, double alpha);
};
} // namespace LineSearch