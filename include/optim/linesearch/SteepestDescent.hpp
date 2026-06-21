#pragma once
#include "Eigen/Dense"
#include "optim/linesearch/LineSearchBase.hpp"
#include "optim/OptimizationResult.hpp"

namespace optim::logger {
class Logger;
}

namespace optim { class DifferentiableFunction; }

namespace optim::linesearch {
class SteepestDescent : public LineSearchBase<optim::DifferentiableFunction> {
  public:
    SteepestDescent(SearchStrategy search_strategy = SearchStrategy::ARMIJO,
                    int max_iterations = 1000, optim::ConvergenceCriteria criteria = {},
                    std::shared_ptr<optim::logger::Logger> logger = nullptr);

    optim::OptimizationResult optimize(const optim::DifferentiableFunction &f,
                                       const Eigen::VectorXd &x) override;
};
} // namespace optim::linesearch
