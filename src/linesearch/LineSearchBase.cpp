#include "optim/linesearch/LineSearchBase.hpp"
#include "optim/linesearch/ArmijoBacktracking.hpp"
#include "optim/linesearch/SearchStrategyBase.hpp"
#include "optim/linesearch/StrongWolfe.hpp"
#include "optim/logger/Logger.hpp"
#include <stdexcept>

using namespace optim::linesearch;

LineSearchBase::LineSearchBase(SearchStrategy strategy, int max_iterations,
                               optim::ConvergenceCriteria criteria,
                               std::shared_ptr<optim::logger::Logger> logger)
    : OptimizerBase(max_iterations, criteria, logger) {
    setStrategy(strategy);
}

void LineSearchBase::setStrategy(SearchStrategy s) {
    switch (s) {
    case SearchStrategy::ARMIJO:
        setConfig(ArmijoConfig{}); // default Armijo
        break;
    case SearchStrategy::STRONG_WOLFE:
        setConfig(WolfeConfig{}); // default Wolfe
        break;
    default:
        throw std::invalid_argument("[LineSearchBase] Undefined strategy.");
    }
}

void LineSearchBase::setConfig(const ArmijoConfig &cfg) {
    search_strategy_ = std::make_shared<ArmijoBacktracking>(cfg, logger_);
}

void LineSearchBase::setConfig(const WolfeConfig &cfg) {
    search_strategy_ = std::make_shared<StrongWolfe>(cfg, logger_);
}

void LineSearchBase::setLogger(std::shared_ptr<optim::logger::Logger> logger) {
    OptimizerBase::setLogger(logger);
    search_strategy_->setLogger(logger);
}