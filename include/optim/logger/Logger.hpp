#pragma once
#include <Eigen/Dense>
#include <string>

namespace optim::logger {
enum class Verbosity { QUIET, ERROR, WARN, INFO };

class Logger {
  public:
    Logger(Verbosity v);
    virtual ~Logger() = default;

    void setVerbosity(Verbosity level);
    Verbosity getVerbosity() const;

    virtual void log(const std::string &message) = 0;
    virtual void logIteration(const IterationInfo &it) = 0;

    bool shouldLog(Verbosity msgLevel) const { return msgLevel <= verbosity_; }

  protected:
    std::string toString(Verbosity level) const;
    Verbosity verbosity_;
};
} // namespace optim::logger