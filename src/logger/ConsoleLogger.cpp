#include "optimization/ConsoleLogger.hpp"
#include <iomanip>
#include <iostream>

void ConsoleLogger::logIteration(int iter, const Eigen::VectorXd &x, const Eigen::VectorXd &grad,
                                 double f_val, const Eigen::VectorXd &step) {
    std::cout << std::fixed << std::setprecision(6) << std::setw(6) << "Iter:" << " "
              << std::setw(3) << iter << "  " << std::setw(6) << "f(x):" << " " << std::setw(10)
              << f_val << "  " << std::setw(8) << "||grad||:" << " " << std::setw(10) << grad.norm()
              << "  " << std::setw(6) << "step:" << " " << std::setw(8) << step.transpose() << "  "
              << std::setw(3) << "x:" << " " << x.transpose() << std::endl;
}
