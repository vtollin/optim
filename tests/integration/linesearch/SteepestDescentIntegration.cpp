#include "Beale.hpp"
#include "ConstantFunction.hpp"
#include "ConvexQuadratic.hpp"
#include "Himmelblau.hpp"
#include "Rosenbrock.hpp"
#include "Wood.hpp"
#include "optim/linesearch/ArmijoBacktracking.hpp"
#include "optim/linesearch/SteepestDescent.hpp"
#include "optim/linesearch/StrongWolfe.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>

using namespace optim::linesearch;
using namespace optim;

TEST(SteepestDescentIntegration, ConvexQuadArmijo) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 5.0, -3.0;

    ArmijoConfig config;
    config.alpha_init = 0.8;
    config.c1 = 1e-4;

    SteepestDescent optimizer(std::make_unique<ArmijoBacktracking>(config), /*max_iters=*/1000, {});

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.f_val, 0.0, 1e-8);
    EXPECT_NEAR(result.x_opt(0), 0.0, 1e-4);
    EXPECT_NEAR(result.x_opt(1), 0.0, 1e-4);
}

TEST(SteepestDescentIntegration, RosenbrockArmijo) {
    Rosenbrock f;
    Eigen::Vector2d x0;
    x0 << -1.2, 1.0;

    ArmijoConfig config;
    config.alpha_init = 1.0;
    config.c1 = 1e-4;

    SteepestDescent optimizer(std::make_unique<ArmijoBacktracking>(config), /*max_iters=*/7000, {});

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-5);
    EXPECT_NEAR(result.x_opt(0), 1.0, 1e-3);
    EXPECT_NEAR(result.x_opt(1), 1.0, 1e-3);
}

TEST(SteepestDescentIntegration, HimmelblauArmijo) {
    Himmelblau f;
    Eigen::Vector2d x0;
    x0 << 0.5, 0.5;

    SteepestDescent optimizer(StepLengthMethod::ARMIJO, /*max_iters=*/10000, {});

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-7);
    EXPECT_NEAR(result.x_opt(0), 3.0, 1e-5);
    EXPECT_NEAR(result.x_opt(1), 2.0, 1e-5);
}

TEST(SteepestDescentIntegration, BealeArmijo) {
    Beale f;
    Eigen::Vector2d x0;
    x0 << 1.2, 1.2;

    SteepestDescent optimizer(StepLengthMethod::ARMIJO, /*max_iters=*/20000, {});

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-6);
    EXPECT_NEAR(result.x_opt(0), 3.0, 1e-2);
    EXPECT_NEAR(result.x_opt(1), 0.5, 1e-3);
}

TEST(SteepestDescentIntegration, WoodArmijo) {
    Wood f;
    Eigen::Vector4d x0;
    x0 << -3.0, -1.0, -3.0, -1.0;

    SteepestDescent optimizer(StepLengthMethod::ARMIJO, /*max_iters=*/10000, {});

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-6);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(result.x_opt(i), 1.0, 1e-3);
    }
}

TEST(SteepestDescentIntegration, ConvexQuadWolfe) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 5.0, -3.0;

    SteepestDescent optimizer(StepLengthMethod::STRONG_WOLFE, /*max_iters=*/1000, {});

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.f_val, 0.0, 1e-10);
    EXPECT_NEAR(result.x_opt.norm(), 0.0, 1e-7);
}

TEST(SteepestDescentIntegration, RosenbrockWolfe) {
    Rosenbrock f;
    Eigen::Vector2d x0;
    x0 << -1.2, 1.0;

    SteepestDescent optimizer(StepLengthMethod::STRONG_WOLFE, 15000, {});

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-3);
    EXPECT_NEAR(result.x_opt(0), 1.0, 1e-2);
    EXPECT_NEAR(result.x_opt(1), 1.0, 1e-2);
}

TEST(SteepestDescentIntegration, WoodWolfe) {
    Wood f;
    Eigen::Vector4d x0;
    x0 << -3.0, -1.0, -3.0, -1.0;

    SteepestDescent optimizer(StepLengthMethod::STRONG_WOLFE, /*max_iters=*/10000, {});

    auto result = optimizer.optimize(f, x0);
    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-3);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(result.x_opt(i), 1.0, 1e-2);
    }
}
