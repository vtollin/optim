#include "optim/linesearch/LineSearchBase.hpp"
#include "optim/linesearch/ArmijoBacktracking.hpp"
#include "optim/linesearch/StepLengthPolicy.hpp"
#include "optim/linesearch/StrongWolfe.hpp"
#include "optim/logger/Logger.hpp"
#include "optim/Functions.hpp"
#include <stdexcept>

using namespace optim::linesearch;

template <typename FuncType>
LineSearchBase<FuncType>::LineSearchBase(SearchStrategy strategy, int max_iterations,
                                         optim::ConvergenceCriteria criteria,
                                         std::shared_ptr<optim::logger::Logger> logger)
    : OptimizerBase(max_iterations, criteria, logger) {
    setStrategy(strategy);
}

template <typename FuncType>
void LineSearchBase<FuncType>::setStrategy(SearchStrategy s) {
    switch (s) {
    case SearchStrategy::ARMIJO:
        setConfig(ArmijoConfig{});
        break;
    case SearchStrategy::STRONG_WOLFE:
        setConfig(WolfeConfig{});
        break;
    default:
        throw std::invalid_argument("[LineSearchBase] Undefined strategy.");
    }
}

template <typename FuncType>
void LineSearchBase<FuncType>::setConfig(const ArmijoConfig &cfg) {
    step_length_policy_ = std::make_unique<ArmijoBacktracking>(cfg, logger_);
}

template <typename FuncType>
void LineSearchBase<FuncType>::setConfig(const WolfeConfig &cfg) {
    step_length_policy_ = std::make_unique<StrongWolfe>(cfg, logger_);
}

template <typename FuncType>
void LineSearchBase<FuncType>::setLogger(std::shared_ptr<optim::logger::Logger> logger) {
    OptimizerBase::setLogger(logger);
    step_length_policy_->setLogger(logger);
}

template class LineSearchBase<optim::DifferentiableFunction>;
template class LineSearchBase<optim::TwiceDifferentiableFunction>;
