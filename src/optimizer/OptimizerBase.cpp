#include "optim/OptimizerBase.hpp"
#include <memory>

using namespace optim;

OptimizerBase::OptimizerBase(int max_iterations, ConvergenceCriteria criteria,
                             optim::logger::Logger *logger)
    : max_iterations_(max_iterations), criteria_(criteria), logger_(logger) {
}

void OptimizerBase::setLogger(optim::logger::Logger *logger) {
    logger_ = logger;
}
