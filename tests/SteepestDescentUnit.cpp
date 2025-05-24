#include "ConstantStepSearch.hpp"
#include "ConvexQuadratic.hpp"
#include "Rosenbrock.hpp"
#include "optimization/ConsoleLogger.hpp"
#include "optimization/OptimizationResult.hpp"
#include "optimization/Optimizer/SteepestDescent.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

using namespace Optimizer;

TEST(SteepestDescentUnit, SimpleConvex) {
    ConvexQuadratic f;
    auto ls = std::make_shared<ConstantStepSearch>(1.0);
    Eigen::VectorXd x0(2);
    x0 << 2.0, 2.0;
    SteepestDescent sd(ls, 10, 1e-6);

    OptimizationResult result = sd.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 1);
}

TEST(SteepestDescentUnit, InvalidVector) {
    ConvexQuadratic f;
    auto ls = std::make_shared<ConstantStepSearch>(1.0);
    Eigen::VectorXd x0(3);
    x0 << 2.0, 2.0, 2.0;
    SteepestDescent sd(ls, 10, 1e-6);

    EXPECT_THROW(sd.optimize(f, x0), std::invalid_argument);
}

TEST(SteepestDescentUnit, ZeroStep) {
    ConvexQuadratic f;
    auto ls = std::make_shared<ConstantStepSearch>(0.0);
    Eigen::VectorXd x0(2);
    x0 << 2.0, 2.0;
    SteepestDescent sd(ls, 10, 1e-6);

    OptimizationResult result = sd.optimize(f, x0);

    EXPECT_FALSE(result.converged);
    std::string msg = "Search strategy returned 0 step.";
    EXPECT_EQ(result.message, msg);
}