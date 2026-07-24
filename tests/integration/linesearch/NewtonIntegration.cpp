#include "ConvexQuadratic.hpp"
#include "Himmelblau.hpp"
#include "RankDeficientQuad.hpp"
#include "Rosenbrock.hpp"
#include "Saddle.hpp"
#include "Wood.hpp"
#include "optim/linesearch/Newton.hpp"
#include "optim/linesearch/StepLengthMethod.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <gtest/gtest.h>

using namespace optim::linesearch;
using namespace optim;

struct NewtonTestObserver : public NewtonObserver {
    std::vector<CholeskyDiagnostics> choleskyUpdates;
    void onModifiedCholesky(const CholeskyDiagnostics &info) override {
        choleskyUpdates.push_back(info);
    }
};

// Simple, convex problem with ArmijoBacktracking. By using exact Hessian, Newton steps
// directly to minimum in one iteration. Ensures modified Cholesky doesn't fire spuriously.
TEST(NewtonIntegration, ConvexQuad) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 5.0, -3.0;

    Newton optimizer;
    NewtonTestObserver obs;
    optimizer.setNewtonObserver(&obs);
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.f_val, 0.0);
    EXPECT_EQ(result.x_opt.norm(), 0.0);
    EXPECT_EQ(result.iterations, 1.0);
    EXPECT_EQ(obs.choleskyUpdates[0].max_shift, 0.0);
}

// Simple function with indefinite hessian. Ensures that modified Cholesky factorization executes
// and pulls optimizer away from saddle point at (0, 0).
TEST(NewtonIntegration, SimpleIndefinite) {
    Saddle f;
    Eigen::VectorXd x0(2);
    x0 << 0.0, 0.5;

    Newton optimizer(StepLengthMethod::ARMIJO, 5);
    NewtonTestObserver obs;
    optimizer.setNewtonObserver(&obs);
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_LE(result.f_val, -2.0);
    EXPECT_GE(result.x_opt.norm(), 1.0);
    EXPECT_TRUE(std::all_of(obs.choleskyUpdates.begin(), obs.choleskyUpdates.end(),
                            [](const CholeskyDiagnostics &u) { return u.max_shift > 0.0; }));
}

// Start near saddle at ~(-0.270, -0.923) in nonconvex Himmelblau function. Stresses modified
// Cholesky factorization to reach minimum at (3.0, 2.0).
TEST(NewtonIntegration, HimmelblauSaddle) {
    Himmelblau f;
    Eigen::VectorXd x0(2);
    x0 << -0.270, -0.923;

    Newton optimizer;
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_LT(result.f_val, 1e-8);
    EXPECT_NEAR(result.x_opt(0), 3.0, 1e-8);
    EXPECT_NEAR(result.x_opt(1), 2.0, 1e-8);
}

// Near the minimum the Hessian is positive definite, so Newton converges
// quadratically (~75 iters), in contrast with SteepestDescent which took 2000+ iterations to
// get within 5e-3 of the minimum.
TEST(NewtonIntegration, RosenbrockValley) {
    Rosenbrock f;
    Eigen::VectorXd x0(2);
    x0 << -1.2, 1.0;

    Newton optimizer;
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_LT(result.f_val, 1e-8);
    EXPECT_NEAR(result.x_opt(0), 1.0, 1e-8);
    EXPECT_NEAR(result.x_opt(1), 1.0, 1e-8);
    EXPECT_LT(result.iterations, 100);
}

// Higher-dimensional (4D) nonconvex problem: verifies modified Cholesky and the Newton step
// generalize past the 2D case.
TEST(NewtonIntegration, WoodHigherDim) {
    Wood f;
    Eigen::VectorXd x0(4);
    x0 << -3.0, -1.0, -3.0, -1.0;

    Newton optimizer;
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-8);
    EXPECT_NEAR(result.x_opt(0), 1.0, 1e-6);
    EXPECT_NEAR(result.x_opt(1), 1.0, 1e-6);
    EXPECT_NEAR(result.x_opt(2), 1.0, 1e-6);
    EXPECT_NEAR(result.x_opt(3), 1.0, 1e-6);
}

// Newton's modified Cholesky handles singular hessian (diag(2, 0)) and steps directly along x1.
TEST(NewtonIntegration, SingularHessian) {
    RankDeficientQuad f; // f(x1, x2) = x1^2
    Eigen::VectorXd x0(2);
    x0 << 1.0, 1.0;

    Newton optimizer;
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_LT(result.f_val, 1e-8);
    EXPECT_NEAR(result.x_opt(0), 0.0, 1e-8);
    EXPECT_EQ(result.x_opt(1), 1.0);
}
