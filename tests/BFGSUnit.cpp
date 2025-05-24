#include "ConstantStepSearch.hpp"
#include "ConvexQuadratic.hpp"
#include "optimization/ConsoleLogger.hpp"
#include "optimization/OptimizationResult.hpp"
#include "optimization/Optimizer/BFGS.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

using namespace Optimizer;

TEST(BFGSUnit, SimpleConvex) {
    ConvexQuadratic f;
    auto ls = std::make_shared<ConstantStepSearch>(1.0);
    Eigen::VectorXd x0(2);
    x0 << 2.0, 2.0;
    BFGS sd(ls, 10, 1e-6);

    OptimizationResult result = sd.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 1);
}

TEST(BFGSUnit, InvalidVector) {
    ConvexQuadratic f;
    auto ls = std::make_shared<ConstantStepSearch>(1.0);
    Eigen::VectorXd x0(3);
    x0 << 2.0, 2.0, 2.0;
    BFGS sd(ls, 10, 1e-6);

    EXPECT_THROW(sd.optimize(f, x0), std::invalid_argument);
}

TEST(BFGSUnit, ZeroStep) {
    ConvexQuadratic f;
    auto ls = std::make_shared<ConstantStepSearch>(0.0);
    Eigen::VectorXd x0(2);
    x0 << 2.0, 2.0;
    BFGS sd(ls, 10, 1e-6);

    OptimizationResult result = sd.optimize(f, x0);

    EXPECT_FALSE(result.converged);
    std::string msg = "Search strategy returned 0 step.";
    EXPECT_EQ(result.message, msg);
}