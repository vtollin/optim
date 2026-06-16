#include "optim/logger/ConsoleLogger.hpp"
#include <cmath> // for std::isnan
#include <iomanip>
#include <iostream>

using namespace optim::logger;

ConsoleLogger::ConsoleLogger(Verbosity v) : Logger(v) {
}

void ConsoleLogger::log(const std::string &message) {
    std::cout << "[" << toString(getVerbosity()) << "] " << message << '\n';
}

void ConsoleLogger::logIteration(const IterationInfo &info) {
    if (!shouldLog(Verbosity::INFO))
        return;

    std::cout << std::fixed << std::setprecision(6) << "Iter: " << std::setw(4) << info.iter
              << " | f(x): " << std::setw(12) << info.fval << " | ||grad||: " << std::setw(12)
              << info.grad.norm() << " | ||step||: " << std::setw(12) << info.step.norm();

    if (!std::isnan(info.rho)) {
        std::cout << " | rho: " << std::setw(8) << info.rho;
    }
    if (!std::isnan(info.delta)) {
        std::cout << " | delta: " << std::setw(8) << info.delta;
        std::cout << " | " << (info.accepted ? "ACCEPT" : "REJECT");
    }
    // accepted is always set (defaults false)

    std::cout << std::endl;
}
