#include "optim/logger/Logger.hpp"
#include <ostream>
#include <string>

using namespace optim::logger;

Logger::Logger(Verbosity v, std::ostream &os) : verbosity_(v), os_(os) {
}

void Logger::log(Verbosity level, const std::string &msg) const {
    if (level <= verbosity_) {
        os_ << '[' << toString(level) << ']' << msg << '\n';
    }
}

std::string Logger::toString(Verbosity level) const {
    switch (level) {
    case Verbosity::WARN:
        return "WARN";
    case Verbosity::DEBUG:
        return "DEBUG";
    default:
        return "";
    }
}