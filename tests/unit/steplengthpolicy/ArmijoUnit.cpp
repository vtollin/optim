#include "ConvexQuadratic.hpp"
#include "Quadratic1D.hpp"
#include "optim/linesearch/ArmijoBacktracking.hpp"
#include "optim/linesearch/StepLengthPolicy.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <stdexcept>

using namespace optim::linesearch;

// ArmijoBacktracking returns the correct step in 2D quadratic.
TEST(ArmijoUnit, SimpleConvexQuad) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 3.0, 4.0;

    ArmijoBacktracking ls;

    Eigen::VectorXd grad = f.gradient(x0);
    Eigen::VectorXd dir = -grad;

    StepResult res = ls.computeStep(f, x0, dir, grad);
    double alpha = res.alpha;
    StepStatus status = res.status;

    Eigen::VectorXd step = alpha * dir;
    double phi0 = f.evaluate(x0);
    double phi1 = f.evaluate(x0 + step);

    EXPECT_NEAR(step(0), -3.0, 1e-8);
    EXPECT_NEAR(step(1), -4.0, 1e-8);
    EXPECT_LE(phi1, phi0 + 1e-4 * step.dot(grad));
    EXPECT_EQ(status, StepStatus::SUCCESS);
}

// ArmijoBacktracking correctly backtracks from initial overshoot (alpha_init = 100.0).
TEST(ArmijoUnit, ForcedOvershoot) {
    Quadratic1D f;
    Eigen::VectorXd x0(1);
    x0 << -1.0;

    ArmijoConfig cfg;
    cfg.alpha_init = 100.0;
    ArmijoBacktracking ls(cfg);

    Eigen::VectorXd grad = f.gradient(x0);
    Eigen::VectorXd dir = -grad;

    StepResult res = ls.computeStep(f, x0, dir, grad);
    double alpha = res.alpha;
    StepStatus status = res.status;

    Eigen::VectorXd step = alpha * dir;
    double phi0 = f.evaluate(x0);
    double phi1 = f.evaluate(x0 + step);

    EXPECT_LT(alpha, cfg.alpha_init);
    EXPECT_LE(phi1, phi0 + 1e-4 * step.dot(grad));
    EXPECT_EQ(status, StepStatus::SUCCESS);
}

// ArmijoBacktracking returns StepStatus::INADEQUATE when step satisfying sufficient decrease can
// not be found.
TEST(ArmijoUnit, InadequateStepReturn) {
    Quadratic1D f;
    Eigen::VectorXd x0(1);
    x0 << -2.0;

    ArmijoConfig cfg; // pathological config
    cfg.c1 = 0.9;
    cfg.alpha_init = 10.0;
    cfg.max_iters = 2;
    ArmijoBacktracking ls(cfg);

    Eigen::VectorXd grad = f.gradient(x0);
    Eigen::VectorXd dir = -grad;

    StepResult res = ls.computeStep(f, x0, dir, grad);
    double alpha = res.alpha;
    StepStatus status = res.status;

    Eigen::VectorXd step = alpha * dir;
    double phi0 = f.evaluate(x0);
    double phi1 = f.evaluate(x0 + step);

    EXPECT_GT(phi1, phi0 + cfg.c1 * step.dot(grad));
    EXPECT_LT(phi1, phi0);
    EXPECT_EQ(status, StepStatus::INADEQUATE);
}

// ArmijoBacktracking throws a std::runtime_error when passed a non-descent direction.
TEST(ArmijoUnit, NonDescent) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 3.0, 4.0;

    ArmijoBacktracking ls;

    Eigen::VectorXd grad = f.gradient(x0);
    EXPECT_THROW(ls.computeStep(f, x0, grad, grad), std::runtime_error);
}

// ArmijoBacktracking return StepResult{0.0, StepStatus::FAILURE} when alpha_init already below
// alpha_min.
TEST(ArmijoUnit, InitialFailure) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 3.0, 4.0;

    ArmijoConfig cfg;
    cfg.alpha_init = 1e-16;
    ArmijoBacktracking ls(cfg);

    Eigen::VectorXd grad = f.gradient(x0);
    StepResult res = ls.computeStep(f, x0, -grad, grad);
    EXPECT_EQ(res.alpha, 0.0);
    EXPECT_EQ(res.status, StepStatus::FAILURE);
}