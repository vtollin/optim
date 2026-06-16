#include "ConstantFunction.hpp"
#include "ConvexQuadratic.hpp"
#include "IllCondQuad.hpp"
#include "NonConvex1D.hpp"
#include "Quad10.hpp"
#include "Quadratic1D.hpp"
#include "optim/linesearch/StrongWolfe.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <gtest/gtest.h>

using namespace optim::linesearch;

TEST(StrongWolfeUnit, Quad1D) {
    WolfeConfig cfg;
    StrongWolfe ls(cfg);

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

    double phi0_prime = optim::utility::directionalDerivative(f, x0, dir);
    double phi1_prime = optim::utility::directionalDerivative(f, x0 + step, dir);

    EXPECT_LE(std::abs(phi1_prime), -0.9 * phi0_prime);
}

TEST(StrongWolfeUnit, SimpleConvexQuad) {
    WolfeConfig cfg;
    StrongWolfe ls(cfg);
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 3.0, 4.0;

    Eigen::VectorXd grad = f.gradient(x0);
    Eigen::VectorXd dir = -grad;

    double alpha = ls.computeStep(f, x0, dir, grad);
    Eigen::VectorXd step = alpha * dir;

    EXPECT_EQ(step(0), -3.0);
    EXPECT_EQ(step(1), -4.0);

    double phi0 = f.evaluate(x0);
    double phi1 = f.evaluate(x0 + step);
    EXPECT_LE(phi1, phi0 + 1e-4 * step.dot(grad));

    double phi0_prime = optim::utility::directionalDerivative(f, x0, dir);
    double phi1_prime = optim::utility::directionalDerivative(f, x0 + step, dir);

    EXPECT_LE(std::abs(phi1_prime), -0.9 * phi0_prime);
}

TEST(StrongWolfeUnit, IllConditionedQuad) {
    WolfeConfig cfg;
    StrongWolfe ls(cfg);
    IllCondQuad f;
    Eigen::VectorXd x0(2);
    x0 << 1.0, 1.0;

    Eigen::VectorXd grad = f.gradient(x0);
    Eigen::VectorXd dir = -grad;

    double alpha = ls.computeStep(f, x0, dir, grad);
    Eigen::VectorXd step = alpha * dir;

    // zooms in when alpha does not satisfy sufficient decrease
    EXPECT_LT(alpha, 1.0);

    // returned step satisfies Armijo
    double phi0 = f.evaluate(x0);
    double phi1 = f.evaluate(x0 + step);
    EXPECT_LE(phi1, phi0 + 1e-4 * step.dot(grad));

    double phi0_prime = optim::utility::directionalDerivative(f, x0, dir);
    double phi1_prime = optim::utility::directionalDerivative(f, x0 + step, dir);

    EXPECT_LE(std::abs(phi1_prime), -0.9 * phi0_prime);
}

TEST(StrongWolfeTest, SecondZoomCall) {
    Quad10 f;

    Eigen::VectorXd x0(1);
    x0 << 1.0;
    Eigen::VectorXd grad = f.gradient(x0); // = 20
    Eigen::VectorXd dir = -grad;           // = -20

    // configure Wolfe
    WolfeConfig cfg;
    cfg.alpha_init = 0.0999; // between 0.05 and ~0.1
    cfg.rho = 2.0;
    cfg.c1 = 1e-4;
    cfg.c2 = 0.9;
    StrongWolfe ls(cfg);

    // satisfies armijo but not curvature, calls second zoom
    bool armijo = f.evaluate(x0 + cfg.alpha_init * dir) <=
                  f.evaluate(x0) + cfg.c1 * cfg.alpha_init * dir.dot(grad);

    EXPECT_TRUE(armijo);
    // run
    double alpha = ls.computeStep(f, x0, dir, grad);
    Eigen::VectorXd step = alpha * dir;

    //  Ensure it called zoom for alpha < 0.0999;
    EXPECT_GT(alpha, 0.0);
    EXPECT_LT(alpha, cfg.alpha_init);

    // Armijo still holds at this alpha
    double phi0 = f.evaluate(x0);
    double phi1 = f.evaluate(x0 + step);
    EXPECT_LE(phi1, phi0 + cfg.c1 * step.dot(grad));

    // now satisfy the curvature test
    double phi0p = optim::utility::directionalDerivative(f, x0, dir);
    double phi1p = optim::utility::directionalDerivative(f, x0 + step, dir);
    EXPECT_LE(std::abs(phi1p), -cfg.c2 * phi0p);
}
