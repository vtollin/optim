#include "Beale.hpp"
#include "ConstantFunction.hpp"
#include "Himmelblau.hpp"
#include "Rosenbrock.hpp"
#include "SimpleQuadratic.hpp"
#include "Wood.hpp"
#include "optimization/ConsoleLogger.hpp"
#include "optimization/LineSearch/StrongWolfe.hpp"
#include "optimization/Optimizer/SteepestDescent.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>

using namespace LineSearch;
using namespace Optimizer;

TEST(SteepestDescentWolfe, ConvergesConvexQuadratic) {
    SimpleQuadratic f;
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

TEST(SteepestDescentWolfe, QuadraticAtSolution) {
    SimpleQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 0.0, 0.0;
    WolfeConfig config;
    auto ls = std::make_shared<StrongWolfe>(config);
    SteepestDescent optimizer(ls, 1000, 1e-8);
    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 0);
}

TEST(SteepestDescentWolfe, RosenbrockFunction) {
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

TEST(SteepestDescentWolfe, HimmelblauFunction) {
    Himmelblau f;
    Eigen::Vector2d x0;
    x0 << 0.0, 0.0;
    WolfeConfig config;
    auto ls = std::make_shared<StrongWolfe>(config);
    SteepestDescent optimizer(ls, 10000, 1e-8);
    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-7);
    EXPECT_NEAR(result.x_opt(0), 3.0, 1e-5);
    EXPECT_NEAR(result.x_opt(1), 2.0, 1e-5);
}

TEST(SteepestDescentWolfe, BealeFunction) {
    Beale f;
    Eigen::Vector2d x0;
    x0 << 1.2, 1.2;
    WolfeConfig config;
    config.max_iters = 100;
    auto ls = std::make_shared<StrongWolfe>(config);
    SteepestDescent optimizer(ls, 10000, 1e-8);
    auto result = optimizer.optimize(f, x0);
    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-7);

    // global minimizer at (3.0, 0.5), f(3,0.5)=0
    EXPECT_NEAR(result.x_opt(0), 3.0, 1e-5);
    EXPECT_NEAR(result.x_opt(1), 0.5, 1e-5);
}

TEST(SteepestDescentWolfe, WoodFunction) {
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

TEST(SteepestDescentWolfe, FlatFunction) {
    ConstantFunction f;
    Eigen::Vector2d x0;
    x0 << 1.0, 1.0;
    WolfeConfig config;
    auto ls = std::make_shared<StrongWolfe>(config);
    SteepestDescent optimizer(ls, 100, 1e-8);

    auto result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 0);
}

TEST(SteepestDescentWolfe, TerminatesMaxIterations) {
    Rosenbrock f;
    Eigen::Vector2d x0;
    x0 << -1.2, 1.0;
    WolfeConfig config;
    auto ls = std::make_shared<StrongWolfe>(config);
    SteepestDescent optimizer(ls, 10, 1e-30);

    auto result = optimizer.optimize(f, x0);

    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.iterations, 10);
}
