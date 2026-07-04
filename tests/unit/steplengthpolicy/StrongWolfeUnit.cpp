#include "ConstantFunction.hpp"
#include "ConvexQuadratic.hpp"
#include "IllCondQuad.hpp"
#include "Quad10.hpp"
#include "Quadratic1D.hpp"
#include "optim/linesearch/StepLengthPolicy.hpp"
#include "optim/linesearch/StrongWolfe.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <gtest/gtest.h>

using namespace optim::linesearch;

// StrongWolfe returns the correct step in simple quadratic case.
TEST(StrongWolfeUnit, SimpleConvexQuad) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 3.0, 4.0;

    StrongWolfe ls;

    Eigen::VectorXd grad = f.gradient(x0);
    Eigen::VectorXd dir = -grad;

    StepResult res = ls.computeStep(f, x0, dir, grad);
    double alpha = res.alpha;
    StepStatus status = res.status;

    Eigen::VectorXd step = alpha * dir;
    double phi0 = f.evaluate(x0);
    double phi1 = f.evaluate(x0 + step);
    double phi0_prime = f.gradient(x0).dot(dir);
    double phi1_prime = f.gradient(x0 + step).dot(dir);

    EXPECT_EQ(step(0), -3.0);
    EXPECT_EQ(step(1), -4.0);
    EXPECT_LE(phi1, phi0 + 1e-4 * step.dot(grad));
    EXPECT_LE(std::abs(phi1_prime), -0.9 * phi0_prime);
    EXPECT_EQ(status, StepStatus::SUCCESS);
}

// StrongWolfe calls zoom() when step does not satisfy sufficient decrease, and finds a valid
// step.
TEST(StrongWolfeUnit, FirstZoomCall) {
    IllCondQuad f;
    Eigen::VectorXd x0(2);
    x0 << 1.0, 1.0;

    StrongWolfe ls;

    Eigen::VectorXd grad = f.gradient(x0);
    Eigen::VectorXd dir = -grad;

    StepResult res = ls.computeStep(f, x0, dir, grad);
    double alpha = res.alpha;
    StepStatus status = res.status;

    Eigen::VectorXd step = alpha * dir;
    double phi0 = f.evaluate(x0);
    double phi1 = f.evaluate(x0 + step);
    double phi0_prime = f.gradient(x0).dot(dir);
    double phi1_prime = f.gradient(x0 + step).dot(dir);

    EXPECT_LT(alpha, 1.0);
    EXPECT_LE(phi1, phi0 + 1e-4 * step.dot(grad));
    EXPECT_LE(std::abs(phi1_prime), -0.9 * phi0_prime);
    EXPECT_EQ(status, StepStatus::SUCCESS);
}

// StrongWolfe calls zoom() when step satisfies sufficient decrease but not curvature, and
// finds a valid step.
TEST(StrongWolfeUnit, SecondZoomCall) {
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

    StepResult res = ls.computeStep(f, x0, dir, grad);
    double alpha = res.alpha;
    StepStatus status = res.status;

    Eigen::VectorXd step = alpha * dir;

    //  Ensure it called zoom for alpha < 0.0999;
    EXPECT_GT(alpha, 0.0);
    EXPECT_LT(alpha, cfg.alpha_init);

    double phi0 = f.evaluate(x0);
    double phi1 = f.evaluate(x0 + step);
    double phi0p = f.gradient(x0).dot(dir);
    double phi1p = f.gradient(x0 + step).dot(dir);

    EXPECT_LE(phi1, phi0 + cfg.c1 * step.dot(grad));
    EXPECT_LE(std::abs(phi1p), -cfg.c2 * phi0p);
    EXPECT_EQ(status, StepStatus::SUCCESS);
}

// For Quadratic1D from x0=-2 in the steepest descent direction, the curvature condition
// requires alpha in [0.05, 0.95]. Capping alpha_max = 0.04 keeps the geometric expansion
// entirely below that region: Armijo holds at every trial (no zoom is entered), but the
// curvature condition is never reachable. f strictly decreases at the best step found.
TEST(StrongWolfeUnit, InadequateStepReturn) {
    Quadratic1D f;
    Eigen::VectorXd x0(1);
    x0 << -2.0;

    WolfeConfig cfg;
    cfg.alpha_init = 0.01;
    cfg.rho = 2.0;
    cfg.alpha_max = 0.04;
    StrongWolfe ls(cfg);

    Eigen::VectorXd grad = f.gradient(x0);
    Eigen::VectorXd dir = -grad;

    StepResult res = ls.computeStep(f, x0, dir, grad);

    EXPECT_EQ(res.status, StepStatus::INADEQUATE);
    EXPECT_LT(f.evaluate(x0 + res.alpha * dir), f.evaluate(x0));
}

// std::invalid argument is thrown when a non-descent direction is passed to computeStep().
TEST(StrongWolfeUnit, NonDescent) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 3.0, 4.0;

    StrongWolfe ls;

    Eigen::VectorXd grad = f.gradient(x0);
    Eigen::VectorXd dir = -grad;
    EXPECT_THROW(ls.computeStep(f, x0, grad, grad), std::invalid_argument);
}

// alpha_init = 1e-20 is so small that all trial points are numerically indistinguishable
// from x0 in double precision: phi stays constant at phi0 for every evaluation. The stall
// counter trips and computeStep returns FAILURE with alpha == 0.0.
TEST(StrongWolfeUnit, FailureTinyAlphaInit) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 3.0, 4.0;

    WolfeConfig cfg;
    cfg.alpha_init = 1e-20;
    StrongWolfe ls(cfg);

    Eigen::VectorXd grad = f.gradient(x0);
    Eigen::VectorXd dir = -grad;

    StepResult res = ls.computeStep(f, x0, dir, grad);

    EXPECT_EQ(res.status, StepStatus::FAILURE);
    EXPECT_EQ(res.alpha, 0.0);
}