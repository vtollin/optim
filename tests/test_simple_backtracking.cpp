#include "PoorlyScaledQuadratic.hpp"
#include "Rosenbrock.hpp"
#include "SimpleQuadratic.hpp"
#include "optimization/LineSearch/ArmijoBacktracking.hpp"
#include "optimization/OptimizationUtils.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>

using namespace LineSearch;

TEST(SimpleBacktrackingTest, AcceptFirstConvexQuadratic) {
    SimpleQuadratic f;

    ArmijoConfig config;
    config.alpha_init = 0.8;
    config.c = 1e-4;
    config.rho = 0.5;
    ArmijoBacktracking ls(config);
    Eigen::VectorXd x0(2);
    x0 << 1.0, 0.0;

    Eigen::VectorXd grad = f.gradient(x0);

    Eigen::VectorXd step = ls.computeStep(f, x0, -grad, grad);
    EXPECT_EQ(step.norm(), 0.8 * grad.norm());
}

TEST(SimpleBacktrackingTest, DeclineFirstConvexQuadratic) {
    SimpleQuadratic f;

    ArmijoConfig config;
    config.alpha_init = 2.0;
    config.c = 1e-4;
    config.rho = 0.5;
    ArmijoBacktracking ls(config);
    Eigen::VectorXd x0(2);
    x0 << 1.0, 0.0;

    Eigen::VectorXd grad = f.gradient(x0);

    Eigen::VectorXd step = ls.computeStep(f, x0, -grad, grad);

    EXPECT_EQ(step.norm(), grad.norm());
}

TEST(SimpleBacktrackingTest, PoorlyScaledQuad) {
    PoorlyScaledQuadratic f;
    Eigen::Vector2d x0;
    x0 << 2.0, 2.0;
    Eigen::Vector2d grad = f.gradient(x0);
    Eigen::Vector2d direction = -grad;

    ArmijoConfig config;
    config.alpha_init = 1.0;
    config.rho = 0.5;
    config.c = 1e-4;

    ArmijoBacktracking ls(config);
    Eigen::VectorXd step = ls.computeStep(f, x0, direction, grad);

    EXPECT_LT(step.norm(), grad.norm());
    EXPECT_GT(step.norm(), 1e-6 * grad.norm());
}

TEST(SimpleBacktrackingTest, ThrowsOnNonDescent) {
    SimpleQuadratic f;
    Eigen::Vector2d x0;
    x0 << 1.0, 1.0;
    Eigen::Vector2d grad = f.gradient(x0);
    Eigen::Vector2d direction = grad;

    ArmijoConfig config;
    config.alpha_init = 1.0;
    config.rho = 0.5;

    ArmijoBacktracking ls(config);

    EXPECT_THROW(ls.computeStep(f, x0, direction, grad), std::runtime_error);
}

TEST(SimpleBacktrackingTest, ThrowsAlphaMin) {
    SimpleQuadratic f;
    Eigen::Vector2d x0;
    x0 << 1.0, 1.0;
    Eigen::Vector2d grad = f.gradient(x0);
    Eigen::Vector2d direction = grad;

    ArmijoConfig config;
    config.alpha_init = 1.0;
    config.rho = 0.5;
    config.c = 0.99;

    ArmijoBacktracking ls(config);

    EXPECT_THROW(ls.computeStep(f, x0, direction, grad), std::runtime_error);
}

TEST(SimpleBacktrackingTest, NonOptimalRosenbrock) {
    Rosenbrock f;
    Eigen::VectorXd x(2);
    x << -1.2, 1.0;
    Eigen::VectorXd grad = f.gradient(x);
    Eigen::VectorXd direction = -grad;

    ArmijoConfig config;
    config.alpha_init = 1.0;
    config.rho = 0.5;
    config.c = 1e-4;

    ArmijoBacktracking bls(config);
    Eigen::VectorXd step = bls.computeStep(f, x, direction, grad);

    EXPECT_GT(step.norm(), 0.0);
    EXPECT_LE(step.norm(), grad.norm());
}