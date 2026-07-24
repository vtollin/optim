#include "ConvexQuadratic.hpp"
#include "Rosenbrock.hpp"
#include "Wood.hpp"
#include "optim/OptimizationResult.hpp"
#include "optim/trustregion/Dogleg.hpp"
#include "optim/trustregion/SubproblemStatus.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <gtest/gtest.h>

using namespace optim::trustregion;
using optim::OptimizationResult;

struct TestObserver : public IterationObserver {
    std::vector<IterationInfo> iters;
    void onIteration(const IterationInfo &info) override { iters.push_back(info); }
};

// Dogleg takes Newton step converges in one iteration for convex quadratic.
TEST(DoglegIntegration, SimpleQuad) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << -5.0, 4.0;

    Dogleg optimizer;
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x_opt(0), 0.0, 1e-8);
    EXPECT_NEAR(result.x_opt(1), 0.0, 1e-8);
    EXPECT_EQ(result.iterations, 1);
}

// Dogleg takes step along -grad when the unconstrained minimizer along -grad is outside
// of delta. This is the tau in [0, 1] branch.
TEST(DoglegIntegration, CauchyBranch) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 2.0, 2.0;

    TrustRegionConfig cfg;
    cfg.delta_init = 0.01;
    TestObserver obs;
    Dogleg optimizer(100, {}, cfg);
    optimizer.addObserver(&obs);
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x_opt(0), 0.0, 1e-8);
    EXPECT_NEAR(result.x_opt(1), 0.0, 1e-8);

    Eigen::VectorXd p = obs.iters[0].step;
    Eigen::VectorXd g = obs.iters[0].grad;
    double cos_theta = p.dot(g) / (p.norm() * g.norm());
    EXPECT_NEAR(cos_theta, -1.0, 1e-10);
}

// From (-50.0, 25.0) on convex quadratic with delta_init set to 1.0, Dogleg converges in 6
// iterations and the fall back never fires.
TEST(DoglegIntegration, NoFallback) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << -50.0, 25.0;

    TrustRegionConfig cfg;
    cfg.delta_init = 1.0;
    Dogleg optimizer(100, {}, cfg);
    TestObserver obs;
    optimizer.addObserver(&obs);
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x_opt(0), 0.0, 1e-8);
    EXPECT_NEAR(result.x_opt(1), 0.0, 1e-8);
    EXPECT_LE(result.iterations, 10);
    EXPECT_TRUE(std::all_of(obs.iters.begin(), obs.iters.end(), [](const IterationInfo &u) {
        return u.status != SubproblemStatus::NEGATIVECURVATURE;
    }));
}

// Dogleg converges on Rosenbrock in under 30 iterations, and the Cauchy fall back fires to handle
// nonconvexity.
TEST(DoglegIntegration, FallbackRosenbrock) {
    Rosenbrock f;
    Eigen::VectorXd x0(2);
    x0 << -1.2, 1.0;

    Dogleg optimizer;
    TestObserver obs;
    optimizer.addObserver(&obs);
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-8);
    EXPECT_NEAR(result.x_opt(0), 1.0, 1e-8);
    EXPECT_NEAR(result.x_opt(1), 1.0, 1e-8);
    EXPECT_LE(result.iterations, 30);
    EXPECT_TRUE(std::any_of(obs.iters.begin(), obs.iters.end(), [](const IterationInfo &u) {
        return u.status == SubproblemStatus::NEGATIVECURVATURE;
    }));
}

// Dogleg converges immediately when x0 is at the minimum (zero iterations).
TEST(DoglegIntegration, StartAtMinimum) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 0.0, 0.0;

    Dogleg optimizer;
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x_opt(0), 0.0, 1e-8);
    EXPECT_NEAR(result.x_opt(1), 0.0, 1e-8);
    EXPECT_EQ(result.iterations, 0);
}

// Higher-dimensional (4D) nonconvex problem: verifies Dogleg's subproblem solver generalizes past
// the 2D case. Dogleg converges in ~3900 iterations, so max_iterations is raised. This is expected,
// as the indefinite regions of Wood force Dogleg to fall back to the Cauchy step frequently.
TEST(DoglegIntegration, WoodHigherDim) {
    Wood f;
    Eigen::VectorXd x0(4);
    x0 << -3.0, -1.0, -3.0, -1.0;

    Dogleg optimizer(5000);
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-8);
    EXPECT_NEAR(result.x_opt(0), 1.0, 1e-4);
    EXPECT_NEAR(result.x_opt(1), 1.0, 1e-4);
    EXPECT_NEAR(result.x_opt(2), 1.0, 1e-4);
    EXPECT_NEAR(result.x_opt(3), 1.0, 1e-4);
}

// Dogleg returns correctly when max iterations is reached.
TEST(DoglegIntegration, MaxIters) {
    Rosenbrock f;
    Eigen::VectorXd x0(2);
    x0 << -1.2, 1.0;

    Dogleg optimizer(5);
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.reason, optim::StopReason::MAX_ITERS_REACHED);
}