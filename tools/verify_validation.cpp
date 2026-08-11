// Standalone driver that reproduces every (benchmark, method) run reported in the README's
// "Validation" section and prints iterations, converged, stop reason, f*, |f* - f_target|, and x*
// so the numbers can be checked.

#include "Himmelblau.hpp"
#include "Rosenbrock.hpp"
#include "Wood.hpp"

#include "optim/linesearch/BFGS.hpp"
#include "optim/linesearch/Newton.hpp"
#include "optim/linesearch/SteepestDescent.hpp"
#include "optim/trustregion/Dogleg.hpp"
#include "optim/trustregion/MoreSorensen.hpp"
#include "optim/trustregion/SteihaugCG.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

using optim::OptimizationResult;
using optim::StopReason;

namespace {

std::string stopReasonName(StopReason reason) {
    switch (reason) {
    case StopReason::GRADIENT_CONVERGED:
        return "GRADIENT_CONVERGED";
    case StopReason::MAX_ITERS_REACHED:
        return "MAX_ITERS_REACHED";
    case StopReason::STEP_STALLED:
        return "STEP_STALLED";
    case StopReason::F_CHANGE_BELOW_TOL:
        return "F_CHANGE_BELOW_TOL";
    case StopReason::LINE_SEARCH_FAILURE:
        return "LINE_SEARCH_FAILURE";
    }
    return "UNKNOWN";
}

template <typename Optimizer, typename Func>
void run(const std::string &name, Optimizer optimizer, const Func &f, const Eigen::VectorXd &x0,
         double f_target) {
    OptimizationResult result = optimizer.optimize(f, x0);
    double abs_err = std::abs(result.f_val - f_target);

    std::cout << std::left << std::setw(18) << name << std::right << std::setw(7)
              << result.iterations << "   " << std::left << std::setw(6)
              << (result.converged ? "yes" : "no") << std::setw(20) << stopReasonName(result.reason)
              << std::right << std::scientific << std::setprecision(6) << std::setw(16)
              << result.f_val << std::setw(16) << abs_err << "   x* = [" << std::fixed
              << std::setprecision(6) << result.x_opt.transpose() << "]\n";
}

void printHeader(const std::string &title) {
    std::cout << "\n=== " << title << " ===\n";
    std::cout << std::left << std::setw(18) << "Method" << std::right << std::setw(7) << "iters"
              << "   " << std::left << std::setw(6) << "conv" << std::setw(20) << "stop reason"
              << std::right << std::setw(16) << "f*" << std::setw(16) << "|f*-f_target|"
              << "   x*\n";
}

} // namespace

int main() {
    using namespace optim::linesearch;
    using namespace optim::trustregion;

    // Default max_iterations (1000) is nowhere near enough for SteepestDescent on Rosenbrock/Wood.
    // The integration test caps it at 4000 and explicitly expects non-convergence there, so this is
    // a deliberate departure for the README's validation section.
    constexpr int kSteepestDescentMaxIters = 20000;

    {
        Rosenbrock f;
        Eigen::VectorXd x0(2);
        x0 << -1.2, 1.0;
        printHeader("Rosenbrock  (x0 = (-1.2, 1.0), target f* = 0, target x* = (1, 1))");
        run("Steepest Descent", SteepestDescent(StepLengthMethod::ARMIJO, kSteepestDescentMaxIters),
            f, x0, 0.0);
        run("BFGS", BFGS(), f, x0, 0.0);
        run("Newton", Newton(), f, x0, 0.0);
        run("Dogleg", Dogleg(), f, x0, 0.0);
        run("More-Sorensen", MoreSorensen(), f, x0, 0.0);
        run("Steihaug-CG", SteihaugCG(), f, x0, 0.0);
    }

    {
        Himmelblau f;
        Eigen::VectorXd x0(2);
        x0 << -0.270, -0.923;
        printHeader("Himmelblau  (x0 = (-0.27, -0.923), target f* = 0, four symmetric minima)");
        run("Steepest Descent", SteepestDescent(), f, x0, 0.0);
        run("BFGS", BFGS(), f, x0, 0.0);
        run("Newton", Newton(), f, x0, 0.0);
        run("Dogleg", Dogleg(), f, x0, 0.0);
        run("More-Sorensen", MoreSorensen(), f, x0, 0.0);
        run("Steihaug-CG", SteihaugCG(), f, x0, 0.0);
    }

    {
        Wood f;
        Eigen::VectorXd x0(4);
        x0 << -3.0, -1.0, -3.0, -1.0;
        printHeader("Wood  (x0 = (-3, -1, -3, -1), target f* = 0, target x* = (1, 1, 1, 1))");
        run("Steepest Descent", SteepestDescent(StepLengthMethod::ARMIJO, kSteepestDescentMaxIters),
            f, x0, 0.0);
        run("BFGS", BFGS(), f, x0, 0.0);
        run("Newton", Newton(), f, x0, 0.0);
        run("Dogleg", Dogleg(5000), f, x0, 0.0);
        run("More-Sorensen", MoreSorensen(), f, x0, 0.0);
        run("Steihaug-CG", SteihaugCG(), f, x0, 0.0);
    }

    return 0;
}
