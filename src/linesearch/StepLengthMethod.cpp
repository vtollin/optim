#include "optim/linesearch/StepLengthMethod.hpp"
#include "optim/linesearch/ArmijoBacktracking.hpp"
#include "optim/linesearch/StepLengthPolicy.hpp"
#include "optim/linesearch/StrongWolfe.hpp"
#include <stdexcept>

namespace optim::linesearch {
std::unique_ptr<StepLengthPolicy> makeStepLengthPolicy(StepLengthMethod method) {
    switch (method) {
    case StepLengthMethod::ARMIJO:
        return std::make_unique<ArmijoBacktracking>();
    case StepLengthMethod::STRONG_WOLFE:
        return std::make_unique<StrongWolfe>();
    }
    throw std::invalid_argument("[StepLengthPolicy] Unknown step length method.");
}
} // namespace optim::linesearch