#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <string>

namespace optim::logger {
enum class Verbosity { QUIET, WARN, DEBUG };

class Logger {
  public:
    ~Logger() = default;

    explicit Logger(Verbosity v = Verbosity::WARN, std::ostream &os = std::cerr);

    void log(Verbosity level, const std::string &msg) const;

    bool shouldLog(Verbosity level) const { return level <= verbosity_; }

  private:
    std::string toString(Verbosity level) const;
    Verbosity verbosity_;
    std::ostream &os_;
};
} // namespace optim::logger