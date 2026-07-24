#include "ConstantFunction.hpp"
#include "ConvexQuadratic.hpp"
#include "IllCondQuad.hpp"
#include "Rosenbrock.hpp"
#include "optim/linesearch/ArmijoBacktracking.hpp"
#include "optim/linesearch/SteepestDescent.hpp"
#include "optim/linesearch/StrongWolfe.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>

using namespace optim::linesearch;
using namespace optim;

// Simple, convex problem with ArmijoBacktracking
TEST(SteepestDescentIntegration, ConvexQuadArmijo) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 5.0, -3.0;

    SteepestDescent optimizer(StepLengthMethod::ARMIJO);
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.f_val, 0.0, 1e-8);
    EXPECT_NEAR(result.x_opt(0), 0.0, 1e-8);
    EXPECT_NEAR(result.x_opt(1), 0.0, 1e-8);
}

// Simple, convex problem with StrongWolfe
TEST(SteepestDescentIntegration, ConvexQuadWolfe) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 5.0, -3.0;

    SteepestDescent optimizer(StepLengthMethod::STRONG_WOLFE);
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.f_val, 0.0, 1e-8);
    EXPECT_NEAR(result.x_opt(0), 0.0, 1e-8);
    EXPECT_NEAR(result.x_opt(1), 0.0, 1e-8);
}

// Target weakness of SD: ill-conditioned quadratic
TEST(SteepestDescentIntegration, IllConditionedQuad) {
    IllCondQuad f;
    Eigen::VectorXd x0(2);
    x0 << 1.0, 2.0;

    SteepestDescent optimizer;
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.f_val, 0.0, 1e-8);
    EXPECT_NEAR(result.x_opt(0), 0.0, 1e-8);
    EXPECT_NEAR(result.x_opt(1), 0.0, 1e-8);
}

// Target weakness of SD: begin in Rosenbrock valley, which has ill-conditioned curvature.
// Tolerances are loosened to 5e-3 and max iterations is increased to 4000. This is
// characteristic SD weakness.
TEST(SteepestDescentIntegration, RosenbrockValley) {
    Rosenbrock f;
    Eigen::Vector2d x0;
    x0 << -1.2, 1.0;

    SteepestDescent optimizer(StepLengthMethod::ARMIJO, 4000);
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_NEAR(result.f_val, 0.0, 1e-5);
    EXPECT_NEAR(result.x_opt(0), 1.0, 5e-3);
    EXPECT_NEAR(result.x_opt(1), 1.0, 5e-3);
}