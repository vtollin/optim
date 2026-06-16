#include "Beale.hpp"
#include "ConstantFunction.hpp"
#include "ConvexQuadratic.hpp"
#include "Himmelblau.hpp"
#include "Rosenbrock.hpp"
#include "Wood.hpp"
#include "optim/linesearch/SteepestDescent.hpp"
#include "optim/linesearch/ArmijoBacktracking.hpp"
#include "optim/linesearch/StrongWolfe.hpp"
#include "optim/logger/ConsoleLogger.hpp"
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
    config.c = 1e-4;

    SteepestDescent optimizer(SearchStrategy::ARMIJO, /*max_iters=*/1000, {});
    optimizer.setConfig(config);

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.f_val, 0.0, 1e-8);
    EXPECT_NEAR(result.x_opt(0), 0.0, 1e-4);
    EXPECT_NEAR(result.x_opt(1), 0.0, 1e-4);
    std::cout << result.message << std::endl;
}

TEST(SteepestDescentIntegration, RosenbrockArmijo) {
    Rosenbrock f;
    Eigen::Vector2d x0;
    x0 << -1.2, 1.0;

    ArmijoConfig config;
    config.alpha_init = 1.0;
    config.c = 1e-4;

    SteepestDescent optimizer(SearchStrategy::ARMIJO, /*max_iters=*/7000, {});
    optimizer.setConfig(config);

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-5);
    EXPECT_NEAR(result.x_opt(0), 1.0, 1e-3);
    EXPECT_NEAR(result.x_opt(1), 1.0, 1e-3);
    std::cout << result.message << std::endl;
}

TEST(SteepestDescentIntegration, HimmelblauArmijo) {
    Himmelblau f;
    Eigen::Vector2d x0;
    x0 << 0.5, 0.5;

    // default Armijo parameters
    SteepestDescent optimizer(SearchStrategy::ARMIJO, /*max_iters=*/10000, {});

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-7);
    EXPECT_NEAR(result.x_opt(0), 3.0, 1e-5);
    EXPECT_NEAR(result.x_opt(1), 2.0, 1e-5);
    std::cout << result.message << std::endl;
}

TEST(SteepestDescentIntegration, BealeArmijo) {
    Beale f;
    Eigen::Vector2d x0;
    x0 << 1.2, 1.2;

    ArmijoConfig config; // use defaults
    SteepestDescent optimizer(SearchStrategy::ARMIJO, /*max_iters=*/10000, {});
    optimizer.setConfig(config);

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-6);
    EXPECT_NEAR(result.x_opt(0), 3.0, 1e-2);
    EXPECT_NEAR(result.x_opt(1), 0.5, 1e-3);
    std::cout << result.message << std::endl;
}

TEST(SteepestDescentIntegration, WoodArmijo) {
    Wood f;
    Eigen::Vector4d x0;
    x0 << -3.0, -1.0, -3.0, -1.0;

    ArmijoConfig config; // use defaults
    SteepestDescent optimizer(SearchStrategy::ARMIJO, /*max_iters=*/10000, {});
    optimizer.setConfig(config);

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-6);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(result.x_opt(i), 1.0, 1e-3);
    }
    std::cout << result.message << std::endl;
}

TEST(SteepestDescentIntegration, ConvexQuadWolfe) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 5.0, -3.0;

    WolfeConfig config; // use defaults
    SteepestDescent optimizer(SearchStrategy::STRONG_WOLFE, /*max_iters=*/1000, {});
    optimizer.setConfig(config);

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.f_val, 0.0, 1e-10);
    EXPECT_NEAR(result.x_opt.norm(), 0.0, 1e-7);
    std::cout << result.message << std::endl;
}

TEST(SteepestDescentIntegration, RosenbrockWolfe) {
    Rosenbrock f;
    Eigen::Vector2d x0;
    x0 << -1.2, 1.0;

    WolfeConfig config; // use defaults
    SteepestDescent optimizer(SearchStrategy::STRONG_WOLFE, 15000, {});
    optimizer.setConfig(config);

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-3);
    EXPECT_NEAR(result.x_opt(0), 1.0, 1e-2);
    EXPECT_NEAR(result.x_opt(1), 1.0, 1e-2);
    std::cout << result.message << std::endl;
}

TEST(SteepestDescentIntegration, WoodWolfe) {
    Wood f;
    Eigen::Vector4d x0;
    x0 << -3.0, -1.0, -3.0, -1.0;

    auto logger = std::make_shared<optim::logger::ConsoleLogger>(optim::logger::Verbosity::WARN);

    WolfeConfig config; // use defaults
    SteepestDescent optimizer(SearchStrategy::STRONG_WOLFE, /*max_iters=*/10000, {}, logger);
    optimizer.setConfig(config);

    auto result = optimizer.optimize(f, x0);
    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-3);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(result.x_opt(i), 1.0, 1e-2);
    }
    std::cout << result.message << std::endl;
}
