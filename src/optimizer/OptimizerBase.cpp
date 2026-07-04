#include "optim/OptimizerBase.hpp"

using namespace optim;

OptimizerBase::OptimizerBase(int max_iterations, ConvergenceCriteria criteria)
    : max_iterations_(max_iterations), criteria_(criteria) {
}
