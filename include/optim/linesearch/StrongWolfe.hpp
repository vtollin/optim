#pragma once
#include "optim/OptimizerUtility.hpp"
#include "optim/linesearch/StepLengthPolicy.hpp"
#include <Eigen/Dense>

namespace optim {
class DifferentiableFunction;
}

namespace optim::linesearch {
struct WolfeConfig {
    double alpha_init = 1.0;
    double alpha_max = 10.0;
    double rho = 2.0;
    int max_iters = 20;
    int zoom_max_iters = 15;
    double c1 = 1e-4;
    double c2 = 0.9;
};

class StrongWolfe : public StepLengthPolicy {
  public:
    explicit StrongWolfe(const WolfeConfig &config = WolfeConfig{},
                         optim::logger::Logger *logger = nullptr);

    StepResult computeStep(const optim::DifferentiableFunction &f, const Eigen::VectorXd &x,
                           const Eigen::VectorXd &direction,
                           const Eigen::VectorXd &gradient) override;
    void setConfig(const WolfeConfig &cfg);

  private:
    WolfeConfig config_;
    StepResult zoom(double alpha_lo, double alpha_hi, const optim::DifferentiableFunction &f,
                    const Eigen::VectorXd &x, const Eigen::VectorXd &direction, double phi0,
                    double phi_prime0);
    bool isInvalid(double alpha_lo, double alpha_hi, double alpha);
};
} // namespace optim::linesearch
