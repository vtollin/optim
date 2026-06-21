#pragma once
#include "optim/linesearch/LineSearchBase.hpp"
#include "optim/OptimizationResult.hpp"
#include <Eigen/Dense>

namespace optim::logger {
class Logger;
}

namespace optim { class DifferentiableFunction; }

namespace optim::linesearch {
class BFGS : public LineSearchBase<optim::DifferentiableFunction> {
  public:
    // Strong Wolfe is the default: the curvature condition it enforces implies s^T y > 0,
    // which is what makes the BFGS update maintain positive definiteness.
    // Armijo is permitted because Powell damping in updateBFGS provides a fallback.
    // However, convergence may be slower and the theoretical guarantees are weaker.
    BFGS(SearchStrategy search_strategy = SearchStrategy::STRONG_WOLFE, int max_iterations = 1000,
         optim::ConvergenceCriteria criteria = {},
         std::shared_ptr<optim::logger::Logger> logger = nullptr);

    optim::OptimizationResult optimize(const optim::DifferentiableFunction &f,
                                       const Eigen::VectorXd &x0) override;

  private:
    void updateBFGS(Eigen::MatrixXd &H, const Eigen::VectorXd &s, const Eigen::VectorXd &y_k);
};
} // namespace optim::linesearch
