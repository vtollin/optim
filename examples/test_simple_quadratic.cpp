#include "SimpleQuadratic.hpp"
#include "optimization/Backtracking.hpp"
#include "optimization/SteepestDescent.hpp"
#include <iostream>
#include <memory>

int main() {
    SimpleQuadratic f;

    Eigen::VectorXd x0(2);
    x0 << -3.0, 5.0;

    auto line_search = std::make_shared<Backtracking>(0.8, 0.5, 1e-6);

    SteepestDescent optimizer(line_search, 1000, 1e-10);

    OptimizationResult result = optimizer.optimize(f, x0);

    std::cout << "Converged : " << std::boolalpha << result.converged << "\n";
    std::cout << "Solution  : " << result.x_opt.transpose() << "\n";
    std::cout << "f(x)      : " << result.f_val << "\n";
    std::cout << "Iterations: " << result.iterations << "\n";
    std::cout << "Message   : " << result.message << "\n";

    return 0;
}