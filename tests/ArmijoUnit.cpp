#include "ConvexQuadratic.hpp"
#include "IllCondQuad.hpp"
#include "NonConvex1D.hpp"
#include "Quadratic1D.hpp"
#include "optimization/LineSearch/ArmijoBacktracking.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <stdexcept>

using namespace LineSearch;

TEST(ArmijoUnit, Quad1D) {
    ArmijoBacktracking ls;
    Quadratic1D f;
    Eigen::VectorXd x0(1);
    x0 << 1.0;

    Eigen::VectorXd grad = f.gradient(x0);
    Eigen::VectorXd dir = -grad;

    Eigen::VectorXd step = ls.computeStep(f, x0, dir, grad);

    EXPECT_EQ(step(0), -1.0);

    double phi0 = f.evaluate(x0);
    double phi1 = f.evaluate(x0 + step);
    EXPECT_LE(phi1, phi0 + 1e-4 * step.dot(grad));
}

TEST(ArmijoUnit, SimpleConvexQuad) {
    ArmijoBacktracking ls;
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 3.0, 4.0;

    Eigen::VectorXd grad = f.gradient(x0);
    Eigen::VectorXd dir = -grad;

    Eigen::VectorXd step = ls.computeStep(f, x0, dir, grad);

    EXPECT_NEAR(step(0), -3.0, 1e-8);
    EXPECT_NEAR(step(1), -4.0, 1e-8);

    double phi0 = f.evaluate(x0);
    double phi1 = f.evaluate(x0 + step);
    EXPECT_LE(phi1, phi0 + 1e-4 * step.dot(grad));
}

TEST(ArmijoUnit, IllConditionedQuad) {
    ArmijoConfig config;
    config.strategy = ArmijoConfig::TrialStepOpts::GEOMETRIC;
    ArmijoBacktracking ls(config);
    IllCondQuad f;
    Eigen::VectorXd x0(2);
    x0 << 1.0, 1.0;

    Eigen::VectorXd grad = f.gradient(x0);
    Eigen::VectorXd dir = -grad;

    Eigen::VectorXd step = ls.computeStep(f, x0, dir, grad);

    // alpha was rejected
    double alpha = step.norm() / dir.norm();
    EXPECT_LT(alpha, 1.0);

    // returned step satisfies Armijo
    double phi0 = f.evaluate(x0);
    double phi1 = f.evaluate(x0 + step);
    EXPECT_LE(phi1, phi0 + 1e-4 * step.dot(grad));

    // one less reduction failes Armijo
    double alpha_lo = alpha / 0.5;
    double phi_lo = f.evaluate(x0 + alpha_lo * dir);
    EXPECT_GT(phi_lo, phi0 + 1e-4 * alpha_lo * dir.dot(grad));
}

TEST(ArmijoUnit, NonConvex) {
    NonConvex1D f;
    Eigen::VectorXd x0(1);
    x0 << 0.8;

    Eigen::VectorXd grad = f.gradient(x0);
    Eigen::VectorXd dir = -grad;

    ArmijoConfig config;
    config.strategy = ArmijoConfig::TrialStepOpts::GEOMETRIC;
    ArmijoBacktracking ls;

    Eigen::VectorXd step = ls.computeStep(f, x0, dir, grad);

    double phi0 = f.evaluate(x0);
    double phi1 = f.evaluate(x0 + step);
    EXPECT_LE(phi1, phi0 + 1e-4 * step.dot(grad));

    double alpha = step.norm() / dir.norm();

    EXPECT_LT(alpha, 1.0);

    double alpha_lo = alpha / 0.5;
    double phi_lo = f.evaluate(x0 + alpha_lo * dir);
    EXPECT_GT(phi_lo, phi0 + 1e-4 * alpha_lo * dir.dot(grad));
}

TEST(ArmijoUnit, ErrorHandling) {
    ArmijoConfig config;
    config.alpha_init = 1e-20;
    ArmijoBacktracking ls(config);
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 3.0, 4.0;

    Eigen::VectorXd grad = f.gradient(x0);

    EXPECT_THROW(ls.computeStep(f, x0, grad, grad), std::runtime_error);

    Eigen::VectorXd step = ls.computeStep(f, x0, -grad, grad);
    EXPECT_EQ(step.norm(), 0.0);
}