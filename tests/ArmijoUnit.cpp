#include "ConvexQuadratic.hpp"
#include "IllCondQuad.hpp"
#include "NonConvex1D.hpp"
#include "Quadratic1D.hpp"
#include "optim/linesearch/ArmijoBacktracking.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <stdexcept>

using namespace optim::linesearch;

TEST(ArmijoUnit, Quad1D) {
    ArmijoConfig cfg;
    ArmijoBacktracking ls(cfg);
    Quadratic1D f;
    Eigen::VectorXd x0(1);
    x0 << 1.0;

    Eigen::VectorXd grad = f.gradient(x0);
    Eigen::VectorXd dir = -grad;

    double alpha = ls.computeStep(f, x0, dir, grad);
    Eigen::VectorXd step = alpha * dir;

    EXPECT_EQ(step(0), -1.0);

    double phi0 = f.evaluate(x0);
    double phi1 = f.evaluate(x0 + step);
    EXPECT_LE(phi1, phi0 + 1e-4 * step.dot(grad));
}

TEST(ArmijoUnit, SimpleConvexQuad) {
    ArmijoConfig cfg;
    ArmijoBacktracking ls(cfg);
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 3.0, 4.0;

    Eigen::VectorXd grad = f.gradient(x0);
    Eigen::VectorXd dir = -grad;

    double alpha = ls.computeStep(f, x0, dir, grad);
    Eigen::VectorXd step = alpha * dir;

    EXPECT_NEAR(step(0), -3.0, 1e-8);
    EXPECT_NEAR(step(1), -4.0, 1e-8);

    double phi0 = f.evaluate(x0);
    double phi1 = f.evaluate(x0 + step);
    EXPECT_LE(phi1, phi0 + 1e-4 * step.dot(grad));
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

    double alpha = ls.computeStep(f, x0, -grad, grad);
    EXPECT_EQ(alpha, 0.0);
}