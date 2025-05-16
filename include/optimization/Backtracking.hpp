#pragma once
#include "LineSearch.hpp"

class Backtracking : public LineSearch {
  public:
    Backtracking(double alpha_init = 1.0, double rho = 0.5, double c = 1e-4,
                 double alpha_min = 1e-10);

    double chooseStep(ObjectiveFunction &f, const Eigen::VectorXd &x,
                      const Eigen::VectorXd &direction, const Eigen::VectorXd &gradient,
                      double alpha_override = -1.0) override;

  private:
    double alpha_init_;
    double rho_;
    double c_;
    double alpha_min_;
    int max_iters_ = 20;
};