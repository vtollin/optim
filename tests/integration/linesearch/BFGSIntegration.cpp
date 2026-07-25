#include "ConvexQuadratic.hpp"
#include "Himmelblau.hpp"
#include "Rosenbrock.hpp"
#include "Wood.hpp"
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

// Validates BFGS on simple, quadratic function.
TEST(BFGSIntegration, ConvexQuad) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 5.0, -3.0;

    BFGS optimizer;
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.f_val, 0.0, 1e-8);
    EXPECT_NEAR(result.x_opt(0), 0.0, 1e-8);
    EXPECT_NEAR(result.x_opt(1), 0.0, 1e-8);
    EXPECT_LT(result.iterations, 5);
}

// Exercises BFGS on full Rosenbrock run. BFGS converges in under 40 iterations, demonstrating
// reliable Hessian construction in ill-conditioned region.
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

// Validates Powell damping. Start near a saddle of Himmelblau with Armijo line search. Without the
// curvature guarantee of Strong Wolfe line search, the BFGS update can violate s^T y >= 0.2 * s^T H
// s, triggering Powell damping at least once. rho > 0 for all updates verifies that damping
// successfully enforced positive definiteness.
TEST(BFGSIntegration, PowellDampingEngages) {
    Himmelblau f;
    Eigen::VectorXd x0(2);
    x0 << -0.270, -0.923;

    BFGS optimizer(StepLengthMethod::ARMIJO);
    BFGSTestObserver obs;
    optimizer.setBFGSObserver(&obs);
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_TRUE(std::any_of(obs.updates.begin(), obs.updates.end(),
                            [](const BFGSUpdateInfo &u) { return u.was_damped; }));
    EXPECT_TRUE(std::all_of(obs.updates.begin(), obs.updates.end(),
                            [](const BFGSUpdateInfo &u) { return u.rho > 0.0; }));
}

// Verifies that Powell damping does not trigger spuriously.
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

// Higher-dimensional (4D) problem: verifies that BFGS matrix updates generalize past the
// 2D case.
TEST(BFGSIntegration, WoodHigherDim) {
    Wood f;
    Eigen::VectorXd x0(4);
    x0 << -3.0, -1.0, -3.0, -1.0;

    BFGS optimizer;
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-8);
    EXPECT_NEAR(result.x_opt(0), 1.0, 1e-8);
    EXPECT_NEAR(result.x_opt(1), 1.0, 1e-8);
    EXPECT_NEAR(result.x_opt(2), 1.0, 1e-8);
    EXPECT_NEAR(result.x_opt(3), 1.0, 1e-8);
}
