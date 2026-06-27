#include "Beale.hpp"
#include "ConstantFunction.hpp"
#include "ConvexQuadratic.hpp"
#include "Himmelblau.hpp"
#include "Rosenbrock.hpp"
#include "Wood.hpp"
#include "optim/linesearch/ArmijoBacktracking.hpp"
#include "optim/linesearch/Newton.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>

using namespace optim::linesearch;
using namespace optim;

TEST(NewtonIntegration, ConvexQuadArmijo) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 5.0, -3.0;

    ArmijoConfig config;
    config.alpha_init = 0.8;
    Newton optimizer(std::make_unique<ArmijoBacktracking>(config), /*max_iters=*/1000, {});

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.f_val, 0.0, 1e-9);
    EXPECT_NEAR(result.x_opt.norm(), 0.0, 1e-4);
}

TEST(NewtonIntegration, RosenbrockArmijo) {
    Rosenbrock f;
    Eigen::Vector2d x0;
    x0 << -1.2, 1.0;

    ArmijoConfig config;
    config.alpha_init = 1.0;
    Newton optimizer(std::make_unique<ArmijoBacktracking>(config), /*max_iters=*/10000, {});

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-4);
    EXPECT_NEAR(result.x_opt(0), 1.0, 1e-2);
    EXPECT_NEAR(result.x_opt(1), 1.0, 1e-2);
}

TEST(NewtonIntegration, HimmelblauArmijo) {
    Himmelblau f;
    Eigen::Vector2d x0;
    x0 << 0.0, 0.0;

    Newton optimizer(StepLengthMethod::ARMIJO, /*max_iters=*/10000, {});

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-7);
    EXPECT_NEAR(result.x_opt(0), 3.0, 1e-5);
    EXPECT_NEAR(result.x_opt(1), 2.0, 1e-5);
}

TEST(NewtonIntegration, BealeArmijo) {
    Beale f;
    Eigen::Vector2d x0;
    x0 << 1.2, 1.2;

    Newton optimizer(StepLengthMethod::ARMIJO, /*max_iters=*/10000, {});

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-7);
    EXPECT_NEAR(result.x_opt(0), 3.0, 1e-5);
    EXPECT_NEAR(result.x_opt(1), 0.5, 1e-5);
}

TEST(NewtonIntegration, WoodArmijo) {
    Wood f;
    Eigen::Vector4d x0;
    x0 << -3.0, -1.0, -3.0, -1.0;

    Newton optimizer(StepLengthMethod::ARMIJO, /*max_iters=*/10000, {});

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-6);
    EXPECT_NEAR(result.x_opt(0), 1.0, 1e-2);
    EXPECT_NEAR(result.x_opt(1), 1.0, 1e-2);
    EXPECT_NEAR(result.x_opt(2), 1.0, 1e-2);
    EXPECT_NEAR(result.x_opt(3), 1.0, 1e-2);
}
