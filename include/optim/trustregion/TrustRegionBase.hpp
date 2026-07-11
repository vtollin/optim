#pragma once
#include "optim/OptimizationResult.hpp"
#include "optim/OptimizerBase.hpp"
#include <Eigen/Dense>
#include <stdexcept>

namespace optim {
class TwiceDifferentiableFunction;
}

namespace optim::trustregion {

enum class SubproblemStatus { INTERIOR, BOUNDARY, NEGATIVECURVATURE, HARDCASE };
struct SubproblemResult {
    Eigen::VectorXd p;
    SubproblemStatus status;
};

class TrustRegionBase : public optim::OptimizerBase {
  public:
    optim::OptimizationResult optimize(const optim::TwiceDifferentiableFunction &f,
                                       const Eigen::VectorXd &x0);

  protected:
    double delta_;
    double delta_max_;
    double eta_;

    TrustRegionBase(int max_iterations, optim::ConvergenceCriteria criteria, double delta_init,
                    double delta_max, double eta)
        : OptimizerBase(max_iterations, criteria)
        , delta_(delta_init)
        , delta_max_(delta_max)
        , eta_(eta) {
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

    virtual SubproblemResult solveSubproblem(const Eigen::VectorXd &grad, const Eigen::MatrixXd &B,
                                             double delta) = 0;
    double computeRho(const TwiceDifferentiableFunction &f, const Eigen::VectorXd &x,
                      const Eigen::VectorXd &grad, const Eigen::MatrixXd &B,
                      const Eigen::VectorXd &step);

    virtual Eigen::MatrixXd initializeB(const optim::TwiceDifferentiableFunction &f,
                                        const Eigen::VectorXd &x0);
    virtual Eigen::MatrixXd updateB(const optim::TwiceDifferentiableFunction &f,
                                    const Eigen::VectorXd &x);
};
} // namespace optim::trustregion
