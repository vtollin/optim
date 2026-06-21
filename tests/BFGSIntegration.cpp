#include "Beale.hpp"
#include "ConstantFunction.hpp"
#include "ConvexQuadratic.hpp"
#include "Himmelblau.hpp"
#include "Rosenbrock.hpp"
#include "Wood.hpp"
#include "optim/linesearch/BFGS.hpp"
#include "optim/linesearch/ArmijoBacktracking.hpp"
#include "optim/linesearch/StrongWolfe.hpp"
#include "optim/logger/ConsoleLogger.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>

using namespace optim::linesearch;
using namespace optim;

TEST(BFGSIntegration, ConvexQuadArmijo) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 5.0, -3.0;

    ArmijoConfig config;
    config.alpha_init = 0.8;
    config.c1 = 1e-4;

    BFGS optimizer(std::make_unique<ArmijoBacktracking>(config), 1000, {});

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.f_val, 0.0, 1e-10);
    EXPECT_NEAR(result.x_opt.norm(), 0.0, 1e-7);
    EXPECT_EQ(result.iterations, 13);
}

TEST(BFGSIntegration, RosenbrockArmijo) {
    Rosenbrock f;
    Eigen::Vector2d x0;
    x0 << -1.2, 1.0;

    ArmijoConfig config;
    config.alpha_init = 1.0;
    config.c1 = 1e-4;

    BFGS optimizer(std::make_unique<ArmijoBacktracking>(config), 7000, {});

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-6);
    EXPECT_NEAR(result.x_opt(0), 1.0, 1e-5);
    EXPECT_NEAR(result.x_opt(1), 1.0, 1e-5);
}

TEST(BFGSIntegration, HimmelblauArmijo) {
    Himmelblau f;
    Eigen::Vector2d x0;
    x0 << 0.0, 0.0;

    BFGS optimizer(StepLengthMethod::ARMIJO, 10000, {});
    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-7);
    EXPECT_NEAR(result.x_opt(0), 3.584, 1e-3);
    EXPECT_NEAR(result.x_opt(1), -1.848, 1e-3);
}

TEST(BFGSIntegration, BealeArmijo) {
    Beale f;
    Eigen::Vector2d x0;
    x0 << 1.2, 1.2;

    BFGS optimizer(StepLengthMethod::ARMIJO, 10000, {});

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-7);
    EXPECT_NEAR(result.x_opt(0), 3.0, 1e-5);
    EXPECT_NEAR(result.x_opt(1), 0.5, 1e-5);
}

TEST(BFGSIntegration, WoodArmijo) {
    Wood f;
    Eigen::Vector4d x0;
    x0 << -3.0, -1.0, -3.0, -1.0;

    BFGS optimizer(StepLengthMethod::ARMIJO, 10000, {});

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-7);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(result.x_opt(i), 1.0, 1e-5);
    }
}

TEST(BFGSIntegration, ConvexQuadWolfe) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 5.0, -3.0;

    BFGS optimizer(StepLengthMethod::STRONG_WOLFE, 1000, {});

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.f_val, 0.0, 1e-10);
    EXPECT_NEAR(result.x_opt.norm(), 0.0, 1e-7);
}

TEST(BFGSIntegration, RosenbrockWolfe) {
    Rosenbrock f;
    Eigen::Vector2d x0;
    x0 << -1.2, 1.0;

    BFGS optimizer(StepLengthMethod::STRONG_WOLFE, 15000, {});

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-6);
    EXPECT_NEAR(result.x_opt(0), 1.0, 1e-5);
    EXPECT_NEAR(result.x_opt(1), 1.0, 1e-5);
}

TEST(BFGSIntegration, WoodWolfe) {
    Wood f;
    Eigen::Vector4d x0;
    x0 << -3.0, -1.0, -3.0, -1.0;

    BFGS optimizer(StepLengthMethod::STRONG_WOLFE, 10000, {});

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-7);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(result.x_opt(i), 1.0, 1e-5);
    }
}
