#pragma once
#include "optim/OptimizationResult.hpp"
#include "optim/OptimizerBase.hpp"
#include "optim/trustregion/IterationObserver.hpp"
#include "optim/trustregion/SubproblemStatus.hpp"
#include <Eigen/Dense>
#include <optional>
#include <stdexcept>

namespace optim {
class TwiceDifferentiableFunction;
}

namespace optim::trustregion {

struct SubproblemResult {
    Eigen::VectorXd p;
    SubproblemStatus status;
};

struct TrustRegionConfig {
    double eta = 1e-4;
    std::optional<double> delta_max;
    std::optional<double> delta_init;
};

class TrustRegionBase : public optim::OptimizerBase {
  public:
    optim::OptimizationResult optimize(const optim::TwiceDifferentiableFunction &f,
                                       const Eigen::VectorXd &x0);

  protected:
    TrustRegionConfig config_;

    TrustRegionBase(int max_iterations, optim::ConvergenceCriteria criteria,
                    TrustRegionConfig config = {})
        : OptimizerBase(max_iterations, criteria), config_(std::move(config)) {
        if (config_.eta <= 0.0) {
            throw std::invalid_argument("[TrustRegionBase]: eta must be positive");
        }
        if (config_.delta_max.has_value()) {
            if (config_.delta_max <= 0.0) {
                throw std::invalid_argument("[TrustRegionBase]: delta_max must be positive");
            }
        }
        if (config_.delta_init.has_value()) {
            if (config_.delta_init.value() <= 0) {
                throw std::invalid_argument("[TrustRegionBase]: delta_init must be positive");
            }
        }
        if (config_.delta_init.has_value() && config_.delta_max.has_value()) {
            if (config_.delta_init.value() > config_.delta_max.value()) {
                throw std::invalid_argument(
                    "[TrustRegionBase]: delta_init cannot be greater than delta_max");
            }
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

  private:
    std::vector<optim::trustregion::IterationObserver *> observers_;
    void notifyStart() {
        for (auto *o : observers_)
            o->onStart();
    }
    void notifyFinish() {
        for (auto *o : observers_)
            o->onFinish();
    }
    void notifyIteration(const optim::trustregion::IterationInfo &info) {
        for (auto *o : observers_)
            o->onIteration(info);
    }
};
} // namespace optim::trustregion
