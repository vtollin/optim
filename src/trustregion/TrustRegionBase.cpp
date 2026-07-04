#include "optim/trustregion/TrustRegionBase.hpp"
#include "optim/Functions.hpp"
#include "optim/trustregion/QuadraticModel.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <limits>

static const double sqrt_eps = std::sqrt(std::numeric_limits<double>::epsilon());

using namespace optim::trustregion;
using optim::TwiceDifferentiableFunction;

UpdateResult TrustRegionBase::update(const TwiceDifferentiableFunction &f, const QuadraticModel &m,
                                     const Eigen::VectorXd &x, const Eigen::VectorXd &step) {
    double actual = f.evaluate(x) - f.evaluate(x + step);
    double predicted = -m.g.dot(step) - 0.5 * step.dot(m.B * step);
    double rho = 0.0;
    double delta_old = delta_;
    bool accepted = false;
    // guard against division by 0
    if (std::abs(predicted) > std::numeric_limits<double>::epsilon() * (1.0 + std::abs(actual))) {
        rho = actual / predicted;
    }

    if (rho < 0.25) {
        delta_ = 0.25 * delta_;
    } else {
        double step_norm = step.norm();
        double delta_tol = sqrt_eps * (1.0 + std::max(step_norm, delta_));
        if (rho > 0.75 && std::abs(delta_ - step_norm) < delta_tol) {
            delta_ = std::min(2 * delta_, delta_max_);
        }
    }
    if (rho > eta_) {
        accepted = true;
    }
    return UpdateResult{rho, delta_old, accepted};
}
