#pragma once
#include "optim/logger/Logger.hpp"
#include <Eigen/Dense>
#include <string>

namespace optim::logger {
class ConsoleLogger : public Logger {
  public:
    explicit ConsoleLogger(Verbosity v = Verbosity::INFO);

    void log(const std::string &message) override;

    void logIteration(const IterationInfo &info) override;
};
} // namespace optim::logger