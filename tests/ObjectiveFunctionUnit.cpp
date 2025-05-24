#include "ConvexQuadratic.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>

TEST(ObjectiveFunctionUnit, invalidVectorEvaluate) {
    ConvexQuadratic f;
    Eigen::VectorXd x(3);
    x << 0.0, 0.0, 0.0;
    EXPECT_THROW(f.evaluate(x), std::invalid_argument);
}

TEST(ObjectiveFunctionUnit, invalidVectorGradient) {
    ConvexQuadratic f;
    Eigen::VectorXd x(3);
    x << 0.0, 0.0, 0.0;
    EXPECT_THROW(f.gradient(x), std::invalid_argument);
}

TEST(ObjectiveFunctionUnit, NaNVectorEvaluate) {
    ConvexQuadratic f;
    Eigen::VectorXd x(2);
    x.setConstant(std::numeric_limits<double>::quiet_NaN());
    EXPECT_THROW(f.evaluate(x), std::invalid_argument);
}

TEST(ObjectiveFunctionUnit, NaNVectorGradient) {
    ConvexQuadratic f;
    Eigen::VectorXd x(2);
    x.setConstant(std::numeric_limits<double>::quiet_NaN());
    EXPECT_THROW(f.gradient(x), std::invalid_argument);
}