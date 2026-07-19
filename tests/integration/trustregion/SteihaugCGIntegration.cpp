#include "ConvexQuadratic.hpp"
#include "IllCondQuad.hpp"
#include "Rosenbrock.hpp"
#include "Saddle.hpp"
#include "optim/OptimizationResult.hpp"
#include "optim/trustregion/SteihaugCG.hpp"
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

// f is an exact quadratic so the full Newton step lands on the global minimizer in one iteration.
// The step (||p|| = ||x0|| = 0.5) fits in the initial trust region, so the subproblem returns
// INTERIOR.
TEST(SteihaugCGIntegration, ExactSolveInterior) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 0.3, -0.4;

    TestObserver obs;
    SteihaugCG optimizer;
    optimizer.addObserver(&obs);
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x_opt(0), 0.0, 1e-8);
    EXPECT_NEAR(result.x_opt(1), 0.0, 1e-8);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_EQ(obs.iters[0].status, SubproblemStatus::INTERIOR);
}

// Curvature along the first CG direction is positive, but a tiny delta_init means the unconstrained
// CG step overshoots the region, hitting the "step leaves the trust region" boundary check rather
// than the negative-curvature check. The truncated step still points straight down -grad.
TEST(SteihaugCGIntegration, BoundaryOvershoot) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 3.0, -4.0;

    TrustRegionConfig cfg;
    cfg.delta_init = 0.5;
    TestObserver obs;
    SteihaugCG optimizer(1000, {}, cfg);
    optimizer.addObserver(&obs);
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x_opt(0), 0.0, 1e-8);
    EXPECT_NEAR(result.x_opt(1), 0.0, 1e-8);

    EXPECT_EQ(obs.iters[0].status, SubproblemStatus::BOUNDARY);
    EXPECT_NEAR(obs.iters[0].step.norm(), 0.5, 1e-10);
    Eigen::VectorXd p = obs.iters[0].step;
    Eigen::VectorXd g = obs.iters[0].grad;
    double cos_theta = p.dot(g) / (p.norm() * g.norm());
    EXPECT_NEAR(cos_theta, -1.0, 1e-10);
}

// Saddle's Hessian diag(2,-2) is constant and indefinite. From (1,3), d^T*B*d = 8*(1-9) = -64 < 0
// on the very first CG iteration, so non-positive curvature is detected immediately and the
// subproblem returns NEGATIVECURVATURE.
TEST(SteihaugCGIntegration, NegativeCurvatureSaddle) {
    Saddle f;
    Eigen::VectorXd x0(2);
    x0 << 1.0, 3.0;

    TestObserver obs;
    SteihaugCG optimizer;
    optimizer.addObserver(&obs);
    optimizer.optimize(f, x0);

    EXPECT_EQ(obs.iters[0].status, SubproblemStatus::NEGATIVECURVATURE);
    EXPECT_NEAR(obs.iters[0].step.norm(), obs.iters[0].delta, 1e-10);
}

// Full nonconvex run: Rosenbrock's Hessian is indefinite away from the solution, so the
// negative-curvature path engages at least once. The optimizer still converges despite relying
// only on truncated CG.
TEST(SteihaugCGIntegration, RosenbrockConvergence) {
    Rosenbrock f;
    Eigen::VectorXd x0(2);
    x0 << -1.2, 1.0;

    TestObserver obs;
    SteihaugCG optimizer;
    optimizer.addObserver(&obs);
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-8);
    EXPECT_NEAR(result.x_opt(0), 1.0, 1e-6);
    EXPECT_NEAR(result.x_opt(1), 1.0, 1e-6);
    EXPECT_LE(result.iterations, 45);
    EXPECT_TRUE(std::any_of(obs.iters.begin(), obs.iters.end(), [](const IterationInfo &u) {
        return u.status == SubproblemStatus::NEGATIVECURVATURE;
    }));
}