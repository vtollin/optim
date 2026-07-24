#include "ConstantFunction.hpp"
#include "ConvexQuadratic.hpp"
#include "optim/linesearch/SteepestDescent.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>

using namespace optim::linesearch;
using namespace optim;

// This suite tests the shared functionality of LineSearchBase through SteepestDescent.

// Verifies that line-search optimizers return immediately when x0 has zero gradient.
TEST(LineSearchBaseUnit, StartAtMinimum) {
    ConstantFunction f;
    Eigen::VectorXd x0(2);
    x0 << 0.0, 0.0;

    SteepestDescent optimizer;
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.x_opt(0), 0.0);
    EXPECT_EQ(result.x_opt(1), 0.0);
    EXPECT_EQ(result.iterations, 0);
}

// Verifies that line-search optimizers terminate properly when max_iterations is reached.
TEST(LineSearchBaseUnit, MaxItersReached) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 1.0, 1.0;

    SteepestDescent optimizer(StepLengthMethod::ARMIJO, 1);
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.reason, optim::StopReason::MAX_ITERS_REACHED);
}