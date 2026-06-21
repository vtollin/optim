#include "optim/linesearch/LineSearchBase.hpp"
#include "optim/Functions.hpp"
#include "optim/linesearch/ArmijoBacktracking.hpp"
#include "optim/linesearch/StepLengthPolicy.hpp"
#include "optim/linesearch/StrongWolfe.hpp"
#include "optim/logger/Logger.hpp"
#include <stdexcept>

using namespace optim::linesearch;

template <typename FuncType>
LineSearchBase<FuncType>::LineSearchBase(std::unique_ptr<StepLengthPolicy> step_length_policy,
                                         int max_iterations, optim::ConvergenceCriteria criteria,
                                         optim::logger::Logger *logger)
    : OptimizerBase(max_iterations, criteria, logger)
    , step_length_policy_(std::move(step_length_policy)) {
    step_length_policy_->setLogger(logger);
}

template <typename FuncType>
void LineSearchBase<FuncType>::setLogger(optim::logger::Logger *logger) {
    OptimizerBase::setLogger(logger);
    step_length_policy_->setLogger(logger);
}

template class LineSearchBase<optim::DifferentiableFunction>;
template class LineSearchBase<optim::TwiceDifferentiableFunction>;
