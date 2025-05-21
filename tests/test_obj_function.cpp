#include "SimpleQuadratic.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>

TEST(ObjectiveFunctionTest, invalidVectorEvaluate) {
    SimpleQuadratic f;
    Eigen::VectorXd x(3);
    x << 0.0, 0.0, 0.0;
    EXPECT_THROW(f.evaluate(x), std::invalid_argument);
}

TEST(ObjectiveFunctionTest, invalidVectorGradient) {
    SimpleQuadratic f;
    Eigen::VectorXd x(3);
    x << 0.0, 0.0, 0.0;
    EXPECT_THROW(f.gradient(x), std::invalid_argument);
}

TEST(ObjectiveFunctionTest, NaNVectorEvaluate) {
    SimpleQuadratic f;
    Eigen::VectorXd x(2);
    x.setConstant(std::numeric_limits<double>::quiet_NaN());
    EXPECT_THROW(f.evaluate(x), std::invalid_argument);
}

TEST(ObjectiveFunctionTest, NaNVectorGradient) {
    SimpleQuadratic f;
    Eigen::VectorXd x(2);
    x.setConstant(std::numeric_limits<double>::quiet_NaN());
    EXPECT_THROW(f.gradient(x), std::invalid_argument);
}