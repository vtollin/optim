#include "ConvexQuadratic.hpp"
#include "Himmelblau.hpp"
#include "Rosenbrock.hpp"
#include "optim/linesearch/ArmijoBacktracking.hpp"
#include "optim/linesearch/BFGS.hpp"
#include "optim/linesearch/StrongWolfe.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <gtest/gtest.h>

using namespace optim::linesearch;
using namespace optim;

struct BFGSTestObserver : public BFGSObserver {
    std::vector<BFGSUpdateInfo> updates;
    void onBFGSUpdate(const BFGSUpdateInfo &info) override { updates.push_back(info); }
};

// BFGS converges on convex quadratic in a few iterations, indicating that the Hessian approximation
// is being constructed correctly in the simple case.
TEST(BFGSIntegration, ConvexQuad) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 5.0, -3.0;

    BFGS optimizer;
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.f_val, 0.0);
    EXPECT_EQ(result.x_opt(0), 0.0);
    EXPECT_EQ(result.x_opt(1), 0.0);
    EXPECT_LT(result.iterations, 5);
}

// BFGS converges in Rosenbrock valley in under 40 iterations, demonstrating reliable Hessian
// construction in ill-conditioned region.
TEST(BFGSIntegration, RosenbrockValley) {
    Rosenbrock f;
    Eigen::VectorXd x0(2);
    x0 << -1.2, 1.0;

    BFGS optimizer;
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-8);
    EXPECT_NEAR(result.x_opt(0), 1.0, 1e-8);
    EXPECT_NEAR(result.x_opt(1), 1.0, 1e-8);
    EXPECT_LT(result.iterations, 40);
}

// Start near a saddle of Himmelblau with Armijo line search. Without the curvature condition,
// the BFGS update can violate s^T y >= 0.2 * s^T H s, triggering Powell damping at least once.
// rho > 0 for all updates verifies that damping successfully enforced positive definiteness.
// Convergence is not asserted: the purpose is to verify damping behavior, not optimizer outcome.
TEST(BFGSIntegration, PowellDampingEngages) {
    Himmelblau f;
    Eigen::VectorXd x0(2);
    x0 << -0.270, -0.923;

    BFGS optimizer(StepLengthMethod::ARMIJO);
    BFGSTestObserver obs;
    optimizer.setBFGSObserver(&obs);
    optimizer.optimize(f, x0);

    EXPECT_FALSE(obs.updates.empty());
    EXPECT_TRUE(std::any_of(obs.updates.begin(), obs.updates.end(),
                            [](const BFGSUpdateInfo &u) { return u.was_damped; }));
    EXPECT_TRUE(std::all_of(obs.updates.begin(), obs.updates.end(),
                            [](const BFGSUpdateInfo &u) { return u.rho > 0.0; }));
}

// Strong Wolfe enforces the curvature condition s^T y > 0 at every accepted step, so Powell
// damping should never trigger.
TEST(BFGSIntegration, PowellDampingAbsent) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 5.0, -3.0;

    BFGS optimizer; // Strong Wolfe by default
    BFGSTestObserver obs;
    optimizer.setBFGSObserver(&obs);
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_TRUE(std::all_of(obs.updates.begin(), obs.updates.end(),
                            [](const BFGSUpdateInfo &u) { return !u.was_damped; }));
}
