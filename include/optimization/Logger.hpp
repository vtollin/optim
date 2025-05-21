#pragma once
#include <Eigen/Dense>

class Logger {
  public:
    virtual void logIteration(int iter, const Eigen::VectorXd &x, const Eigen::VectorXd &grad,
                              double f_val, const Eigen::VectorXd &step) = 0;

    virtual ~Logger() = default;
};