#include "Beale.hpp"
#include "ConvexQuadratic.hpp"
#include "Himmelblau.hpp"
#include "Rosenbrock.hpp"
#include "Wood.hpp"
#include "optim/OptimizationResult.hpp"
#include "optim/trustregion/MoreSorensen.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>

using namespace optim::trustregion;
using optim::OptimizationResult;

TEST(MoreSoresenIntegration, ConvexQuad) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 3.0, -4.0;

    MoreSorensen optimizer(100, {}, -1.0, 100.0, 0.1, BMatrixConfig::APPROXIMATE);

    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x_opt(0), 0.0, 1e-7);
    EXPECT_NEAR(result.x_opt(1), 0.0, 1e-7);
}

TEST(MoreSorensenIntegration, RosenbrockFunc) {
    Rosenbrock f;
    Eigen::VectorXd x0(2);
    x0 << -1.2, 1.0;

    MoreSorensen optimizer(100, {}, -1.0, 100.0, 0.1, BMatrixConfig::APPROXIMATE);

    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-7);
    EXPECT_NEAR(result.x_opt(0), 1.0, 1e-6);
    EXPECT_NEAR(result.x_opt(1), 1.0, 1e-6);
}

TEST(MoreSorensenIntegration, BealeFunc) {
    Beale f;
    Eigen::VectorXd x0(2);
    x0 << 1.0, 1.0;

    MoreSorensen optimizer(100, {}, -1.0, 100.0, 0.1, BMatrixConfig::EXACT);

    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-7);
    EXPECT_NEAR(result.x_opt(0), 3.0, 1e-6);
    EXPECT_NEAR(result.x_opt(1), 0.5, 1e-6);
}

TEST(MoreSorensenIntegration, HimmelblauFunc) {
    Himmelblau f;
    Eigen::VectorXd x0(2);
    x0 << 0.0, 0.0;

    MoreSorensen optimizer(100, {}, -1.0, 100.0, 0.1, BMatrixConfig::EXACT);

    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-7);
    EXPECT_NEAR(result.x_opt(0), 3.0, 1e-6);
    EXPECT_NEAR(result.x_opt(1), 2.0, 1e-6);
}

TEST(MoreSorensenIntegration, WoodFunc) {
    Wood f;
    Eigen::VectorXd x0(4);
    x0 << -3.0, -1.0, -3.0, -1.0;

    MoreSorensen optimizer(100, {}, -1.0, 100.0, 0.1, BMatrixConfig::APPROXIMATE);

    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.f_val, 1e-7);
    EXPECT_NEAR(result.x_opt(0), 1.0, 1e-5);
    EXPECT_NEAR(result.x_opt(1), 1.0, 1e-5);
    EXPECT_NEAR(result.x_opt(2), 1.0, 1e-5);
    EXPECT_NEAR(result.x_opt(3), 1.0, 1e-5);
}