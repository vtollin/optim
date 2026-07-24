#include "ConvexQuadratic.hpp"
#include "optim/Functions.hpp"
#include "optim/trustregion/TrustRegionBase.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>

using namespace optim::trustregion;
using namespace optim;

// This suite verifies the shared functionality of TrustRegionBase in isolation from any real
// subproblem solver by using TrivialBoundarySolver.

struct TestObserver : public IterationObserver {
    std::vector<IterationInfo> iters;
    void onIteration(const IterationInfo &info) override { iters.push_back(info); }
};

// Always returned the steepest-descent step scaled to the current trust-region boundary.
class TrivialBoundarySolver : public TrustRegionBase {
  public:
    explicit TrivialBoundarySolver(int max_iterations = 100,
                                   optim::ConvergenceCriteria criteria = {},
                                   TrustRegionConfig config = {})
        : TrustRegionBase(max_iterations, criteria, config) {}

  protected:
    SubproblemResult solveSubproblem(const Eigen::VectorXd &grad, const Eigen::MatrixXd &,
                                     double delta) override {
        return {-delta * grad.normalized(), SubproblemStatus::BOUNDARY};
    }
};

// f(x) = 0.5*x^2 (true curvature 1.0), but hessianImpl returns caller-provided reported_curvature
// instead of the true curvature. TrustRegionBase's predicted-decrease model uses whatever
// hessianImpl returns, so this allows a test to force a mismatch between the model's predicted
// decrease and the function's actual decrease.
class IncorrectCurvatureQuad : public optim::TwiceDifferentiableFunction {
  public:
    explicit IncorrectCurvatureQuad(double reported_curvature)
        : optim::TwiceDifferentiableFunction(1), reported_curvature_(reported_curvature) {}

  protected:
    double evaluateImpl(const Eigen::VectorXd &x) const override { return 0.5 * x(0) * x(0); }
    Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const override {
        Eigen::VectorXd grad(1);
        grad(0) = x(0);
        return grad;
    }
    Eigen::MatrixXd hessianImpl(const Eigen::VectorXd &) const override {
        Eigen::MatrixXd H(1, 1);
        H(0, 0) = reported_curvature_;
        return H;
    }

  private:
    double reported_curvature_;
};

// Verifies that trust-region optimizers return immediately when x0 has zero gradient.
TEST(TrustRegionBaseUnit, StartAtMinimum) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 0.0, 0.0;

    TrivialBoundarySolver optimizer;
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.x_opt(0), 0.0);
    EXPECT_EQ(result.x_opt(1), 0.0);
    EXPECT_EQ(result.iterations, 0);
}

// Verifies that trust-region optimizers terminate properly when max_iterations is reached.
TEST(TrustRegionBaseUnit, MaxItersReached) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << -1.2, 1.0;

    TrivialBoundarySolver optimizer(1);
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.reason, optim::StopReason::MAX_ITERS_REACHED);
}

// ConvexQuadratic is an exact quadratic, so the trust-region model's predicted decrease always
// equals the actual decrease (rho = 1.0). Combined with TrivialBoundarySolver, the step status is
// BOUNDARY and rho > 0.75 on every iteration, so delta must double each time, capped at delta_max.
TEST(TrustRegionBaseUnit, RadiusExpansion) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 10.0, 0.0;

    TrustRegionConfig cfg;
    cfg.delta_init = 0.1;
    cfg.delta_max = 1.0;
    TestObserver obs;
    TrivialBoundarySolver optimizer(10, {}, cfg);
    optimizer.addObserver(&obs);
    optimizer.optimize(f, x0);

    EXPECT_NEAR(obs.iters[0].delta, 0.1, 1e-12);
    EXPECT_NEAR(obs.iters[1].delta, 0.2, 1e-12);
    EXPECT_NEAR(obs.iters[2].delta, 0.4, 1e-12);
    EXPECT_NEAR(obs.iters[3].delta, 0.8, 1e-12);
    EXPECT_NEAR(obs.iters[4].delta, 1.0, 1e-12);
    EXPECT_NEAR(obs.iters[5].delta, 1.0, 1e-12);

    for (const auto &it : obs.iters) {
        EXPECT_NEAR(it.rho, 1.0, 1e-9);
        EXPECT_EQ(it.status, SubproblemStatus::BOUNDARY);
        EXPECT_TRUE(it.accepted);
    }
}

// IncorrectCurvatureQuad(-10000) returns a Hessian significantly off from its true curvature
// (1.0), so the trust-region model badly overestimates the decrease of a step. On the first
// iteration, predicted = 5001.0, actual = 0.5, and rho = 0.5 / 5001.0 < eta (1e-4), so the step is
// rejected and delta contracts to 0.25. On the second iteration, predicted = 312.75, actual =
// 0.21875, and rho = 0.21875 / 3.1275 ~= 6.994e-4 which is greater than eta but less than 0.25, so
// the step is accepted (x moves to 0.75) and delta contracts again.
TEST(TrustRegionBaseUnit, StepRejectionAndRadiusContraction) {
    IncorrectCurvatureQuad f(-10000.0);
    Eigen::VectorXd x0(1);
    x0 << 1.0;

    TrustRegionConfig cfg;
    cfg.delta_init = 1.0;
    TestObserver obs;
    TrivialBoundarySolver optimizer(2, {}, cfg);
    optimizer.addObserver(&obs);
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_NEAR(obs.iters[0].rho, 0.5 / 5001.0, 1e-12);
    EXPECT_FALSE(obs.iters[0].accepted);
    EXPECT_EQ(obs.iters[0].x(0), 1.0);

    EXPECT_NEAR(obs.iters[1].delta, 0.25 * obs.iters[0].delta, 1e-12);
    EXPECT_EQ(obs.iters[1].x(0), 1.0);
    EXPECT_NEAR(obs.iters[1].rho, 0.21875 / 312.75, 1e-12);
    EXPECT_TRUE(obs.iters[1].accepted);
    EXPECT_GT(obs.iters[1].rho, 1e-4);
    EXPECT_LT(obs.iters[1].rho, 0.25);
    EXPECT_EQ(result.x_opt(0), 0.75);
}
