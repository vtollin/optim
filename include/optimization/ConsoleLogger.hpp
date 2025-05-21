#include "optimization/Logger.hpp"
#include <Eigen/Dense>

class ConsoleLogger : public Logger {
  public:
    void logIteration(int iter, const Eigen::VectorXd &x, const Eigen::VectorXd &grad, double f_val,
                      const Eigen::VectorXd &step) override;
};