#include "ConvexQuadratic.hpp"
#include "IllCondQuad.hpp"
#include "Rosenbrock.hpp"
#include "Wood.hpp"
#include "optim/Functions.hpp"
#include "optim/OptimizationResult.hpp"
#include "optim/trustregion/MoreSorensen.hpp"
#include "optim/trustregion/SubproblemStatus.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <gtest/gtest.h>

using namespace optim::trustregion;
using optim::OptimizationResult;

struct TestObserver : public IterationObserver {
    std::vector<IterationInfo> iters;
    void onIteration(const IterationInfo &info) override { iters.push_back(info); }
};

// f(x0, x1, x2) = x2^2 + 2*x2, independent of x0, x1. Hessian is diag(0,0,2): the minimum
// eigenvalue 0 has geometric multiplicity 2. At x0 = (0,0,0) the gradient is (0,0,2), which is
// exactly orthogonal to the eigenspace of lambda1. At lambda = lambda1, the norm of the particular
// solution (1.0) is less than the trust-region radius (2.0), so MoreSorensen's subproblem solver
// must take the hard-case branch and exercise the multiplicity-2 loop.
class HardCaseQuad : public optim::TwiceDifferentiableFunction {
  public:
    HardCaseQuad() : optim::TwiceDifferentiableFunction(3) {}

  protected:
    double evaluateImpl(const Eigen::VectorXd &x) const override {
        return x(2) * x(2) + 2.0 * x(2);
    }

    Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const override {
        Eigen::VectorXd grad(3);
        grad << 0.0, 0.0, 2.0 * x(2) + 2.0;
        return grad;
    }

    Eigen::MatrixXd hessianImpl(const Eigen::VectorXd &x) const override {
        Eigen::VectorXd diag(3);
        diag << 0.0, 0.0, 2.0;
        return diag.asDiagonal();
    }
};

// Ensures that MoreSorensen takes the full Newton step to the global minimizer on a convex
// quadratic when the radius is large enough. The subproblem status on the only iteration is
// INTERIOR.
TEST(MoreSorensenIntegration, InteriorStep) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 3.0, -4.0;

    TrustRegionConfig cfg;
    cfg.eta = 0.1;
    cfg.delta_init = 10.0;
    cfg.delta_max = 100.0;

    MoreSorensen optimizer(100, {}, cfg);
    TestObserver obs;
    optimizer.addObserver(&obs);
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x_opt(0), 0.0, 1e-8);
    EXPECT_NEAR(result.x_opt(1), 0.0, 1e-8);
    EXPECT_EQ(result.iterations, 1);

    ASSERT_FALSE(obs.iters.empty());
    EXPECT_EQ(obs.iters[0].status, SubproblemStatus::INTERIOR);
}

// Verifies that MoreSoresen takes a boundary step when the model function's minimizer is outside of
// the trust-region.
TEST(MoreSorensenIntegration, BoundaryStep) {
    IllCondQuad f;
    Eigen::VectorXd x0(2);
    x0 << -1.0, -1.0;

    TrustRegionConfig cfg;
    cfg.delta_init = 0.05;
    cfg.delta_max = 100.0;

    MoreSorensen optimizer(100, {}, cfg);
    TestObserver obs;
    optimizer.addObserver(&obs);
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x_opt(0), 0.0, 1e-8);
    EXPECT_NEAR(result.x_opt(1), 0.0, 1e-8);

    EXPECT_EQ(obs.iters[0].status, SubproblemStatus::BOUNDARY);
}

// Exercises More-Sorensen on a full nonconvex run on Rosenbrock.
TEST(MoreSorensenIntegration, RosenbrockValley) {
    Rosenbrock f;
    Eigen::VectorXd x0(2);
    x0 << -1.2, 1.0;

    MoreSorensen optimizer;
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-8);
    EXPECT_NEAR(result.x_opt(0), 1.0, 1e-8);
    EXPECT_NEAR(result.x_opt(1), 1.0, 1e-8);
}

// Verifies that MoreSorensen's subproblem solver generalizes to a higher-dimensional (4D)
// nonconvex problem.
TEST(MoreSorensenIntegration, WoodHigherDim) {
    Wood f;
    Eigen::VectorXd x0(4);
    x0 << -3.0, -1.0, -3.0, -1.0;

    MoreSorensen optimizer;
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-8);
    EXPECT_NEAR(result.x_opt(0), 1.0, 1e-8);
    EXPECT_NEAR(result.x_opt(1), 1.0, 1e-8);
    EXPECT_NEAR(result.x_opt(2), 1.0, 1e-8);
    EXPECT_NEAR(result.x_opt(3), 1.0, 1e-8);
}

// Exercises the hard-case branch using the function described above. The particular solution at
// lambda = lambda1 is the 1D minimizer in x2. The hard case adds tau * z1 the ensure that that step
// lies on the trust-region boundary.
TEST(MoreSorensenIntegration, HardCaseMultiplicityTwo) {
    HardCaseQuad f;
    Eigen::VectorXd x0(3);
    x0 << 0.0, 0.0, 0.0;

    TrustRegionConfig cfg;
    cfg.delta_init = 2.0;
    cfg.delta_max = 100.0;

    MoreSorensen optimizer(50, {}, cfg);
    TestObserver obs;
    optimizer.addObserver(&obs);
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_NEAR(result.f_val, -1.0, 1e-8);
    EXPECT_NEAR(result.x_opt(2), -1.0, 1e-8);

    ASSERT_FALSE(obs.iters.empty());
    EXPECT_EQ(obs.iters[0].status, SubproblemStatus::HARDCASE);
    EXPECT_NEAR(obs.iters[0].step.norm(), 2.0, 1e-8);
}