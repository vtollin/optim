#pragma once
#include <memory>

namespace optim::linesearch {

enum class StepLengthMethod { ARMIJO, STRONG_WOLFE };

class StepLengthPolicy;

std::unique_ptr<StepLengthPolicy> makeStepLengthPolicy(StepLengthMethod method);

} // namespace optim::linesearch