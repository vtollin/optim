#include "optim/logger/Logger.hpp"
#include <string>

using namespace optim::logger;

Logger::Logger(Verbosity v) : verbosity_(v) {
}

Verbosity Logger::getVerbosity() const {
    return verbosity_;
}

void Logger::setVerbosity(Verbosity level) {
    verbosity_ = level;
}

std::string Logger::toString(Verbosity level) const {
    switch (level) {
    case Verbosity::ERROR:
        return "ERROR";
    case Verbosity::WARN:
        return "WARN";
    case Verbosity::INFO:
        return "INFO";
    default:
        return "";
    }
}