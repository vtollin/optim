#include "Beale.hpp"
#include "ConstantFunction.hpp"
#include "ConvexQuadratic.hpp"
#include "Himmelblau.hpp"
#include "Rosenbrock.hpp"
#include "Wood.hpp"
#include "optimization/ConsoleLogger.hpp"
#include "optimization/LineSearch/ArmijoBacktracking.hpp"
#include "optimization/LineSearch/StrongWolfe.hpp"
#include "optimization/Optimizer/SteepestDescent.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>

using namespace LineSearch;
using namespace Optimizer;

TEST(SteepestDescentIntegration, ConvexQuadArmijo) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 5.0, -3.0;
    ArmijoConfig config;
    config.alpha_init = 0.8;
    config.rho = 0.5;
    config.c = 1e-4;
    config.strategy = ArmijoConfig::TrialStepOpts::GEOMETRIC;
    auto ls = std::make_shared<ArmijoBacktracking>(config);
    SteepestDescent optimizer(ls, 1000, 1e-8);

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.f_val, 0.0, 1e-10);
    EXPECT_NEAR(result.x_opt.norm(), 0.0, 1e-7);
    EXPECT_EQ(result.iterations, 13);
}

TEST(SteepestDescentIntegration, RosenbrockArmijo) {
    Rosenbrock f;
    Eigen::Vector2d x0;
    x0 << -1.2, 1.0;
    ArmijoConfig config;
    config.alpha_init = 1.0;
    config.rho = 0.5;
    config.c = 1e-4;
    // default to safeguarded interpolation
    auto ls = std::make_shared<ArmijoBacktracking>(config);
    SteepestDescent optimizer(ls, 7000, 1e-6);
    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-6);
    EXPECT_NEAR(result.x_opt(0), 1.0, 1e-5);
    EXPECT_NEAR(result.x_opt(1), 1.0, 1e-5);
}

TEST(SteepestDescentIntegration, HimmelblauArmijo) {
    Himmelblau f;
    Eigen::Vector2d x0;
    x0 << 0.5, 0.5;
    ArmijoConfig config;
    auto ls = std::make_shared<ArmijoBacktracking>(config);
    SteepestDescent optimizer(ls, 10000, 1e-8);
    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-7);
    EXPECT_NEAR(result.x_opt(0), 3.0, 1e-5);
    EXPECT_NEAR(result.x_opt(1), 2.0, 1e-5);
}

TEST(SteepestDescentIntegration, BealeArmijo) {
    Beale f;
    Eigen::Vector2d x0;
    x0 << 1.2, 1.2; // algorithm does not work fast enough for (1, 1)
    ArmijoConfig config;
    auto ls = std::make_shared<ArmijoBacktracking>(config);
    SteepestDescent optimizer(ls, 10000, 1e-8);

    auto result = optimizer.optimize(f, x0);
    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-7);

    // global minimizer at (3.0, 0.5), f(3,0.5)=0
    EXPECT_NEAR(result.x_opt(0), 3.0, 1e-5);
    EXPECT_NEAR(result.x_opt(1), 0.5, 1e-5);
}

TEST(SteepsetDescentIntegration, WoodArmijo) {
    Wood f;
    Eigen::Vector4d x0;
    x0 << -3.0, -1.0, -3.0, -1.0; // recommended start for Wood’s
    ArmijoConfig config;
    auto ls = std::make_shared<ArmijoBacktracking>(config);
    SteepestDescent optimizer(ls, 10000, 1e-8);

    auto result = optimizer.optimize(f, x0);
    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-7);

    // global minimizer at (1,1,1,1), f(1,1,1,1)=0
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(result.x_opt(i), 1.0, 1e-5);
    }
}

TEST(SteepestDescentIntegration, ConvexQuadWolfe) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 5.0, -3.0;
    WolfeConfig config;
    auto ls = std::make_shared<StrongWolfe>(config);
    SteepestDescent optimizer(ls, 1000, 1e-8);

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.f_val, 0.0, 1e-10);
    EXPECT_NEAR(result.x_opt.norm(), 0.0, 1e-7);
}

TEST(SteepestDescentIntegration, RosenbrockWolfe) {
    Rosenbrock f;
    Eigen::Vector2d x0;
    x0 << -1.2, 1.0;
    WolfeConfig config;
    auto ls = std::make_shared<StrongWolfe>(config);
    SteepestDescent optimizer(ls, 15000, 1e-6);
    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-6);
    EXPECT_NEAR(result.x_opt(0), 1.0, 1e-5);
    EXPECT_NEAR(result.x_opt(1), 1.0, 1e-5);
}

TEST(SteepestDescentIntegration, WoodWolfe) {
    Wood f;
    Eigen::Vector4d x0;
    x0 << -3.0, -1.0, -3.0, -1.0; // recommended start for Wood’s
    WolfeConfig config;
    auto ls = std::make_shared<StrongWolfe>(config);
    SteepestDescent optimizer(ls, 10000, 1e-6);

    auto result = optimizer.optimize(f, x0);
    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-7);

    // global minimizer at (1,1,1,1), f(1,1,1,1)=0
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(result.x_opt(i), 1.0, 1e-5);
    }
}