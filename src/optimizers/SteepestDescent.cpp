#include "SteepestDescent.hpp"

SteepestDescent::SteepestDescent(
    std::shared_ptr<LineSearch> line_search_strategy,
    int max_iterations,
    double tol
)
    : line_search_(std::move(line_search_strategy)),
      max_iterations_(max_iterations),
      tol_(tol) {}
            
OptimizationResult SteepestDescent::optimize(
    ObjectiveFunction& f,
    const Eigen::VectorXd& x0
) {
    int k = 0; 
    bool converged = false; 
    Eigen::VectorXd x = x0;

    for (; k < max_iterations_; ++k) {
        Eigen::VectorXd grad = f.gradient(x);

        if (grad.size() != x.size()) {
            throw std::logic_error("Gradient dimension mismatch\n"); 
        }

        if (grad.norm() < tol_) {
            converged = true;
            break; 
        }

        Eigen::VectorXd direction = -grad;
        double alpha = line_search_->chooseStep(f, x, direction, grad); 
        x += alpha * direction; 
    }

    return OptimizationResult {
        x,
        f.evaluate(x),
        k,
        converged,
        converged ? "Converged successfully" : "Max iterations reached"
    };
}