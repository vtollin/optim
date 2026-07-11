#include "ConvexQuadratic.hpp"
#include "IllCondQuad.hpp"
#include "LogisticRegressionLoss.hpp"
#include "optim/trustregion/Dogleg.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>

using namespace optim::trustregion;
using optim::OptimizationResult;

TEST(DoglegIntegration, SimpleQuad) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << -3.0, 4.0;
    Dogleg optimizer(100, {}, TrustRegionConfig{0.1, 1000.0});
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x_opt(0), 0.0, 1e-7);
    EXPECT_NEAR(result.x_opt(1), 0.0, 1e-7);
}

TEST(DoglegIntegration, IllConditionedQuadratic) {
    ConvexQuadratic f;
    Eigen::VectorXd x0(2);
    x0 << 2.0, 2.0;
    Dogleg optimizer(100, {}, TrustRegionConfig{0.1, 1000.0});
    OptimizationResult result = optimizer.optimize(f, x0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x_opt(0), 0.0, 1e-7);
    EXPECT_NEAR(result.x_opt(1), 0.0, 1e-7);
}

TEST(DoglegIntegration, LogisticRegression) {
    int n = 10; // number of samples
    int d = 3;  // dimensionality

    Eigen::MatrixXd X(n, d);
    X << 1, 2, 3, 2, 1, 3, 3, 4, 1, 4, 3, 2, 1, 0, 2, 0, 1, 1, 3, 3, 3, 2, 2, 2, 1, 4, 3, 4, 1, 0;

    Eigen::VectorXd y(n);
    y << 1, -1, 1, -1, 1, -1, 1, -1, 1, -1;

    double lambda = 0.1; // regularization strength

    LogisticRegressionLoss f(X, y, lambda);
    Eigen::VectorXd w0 = Eigen::VectorXd::Zero(d); // start at zero

    Dogleg optimizer(100, {}, TrustRegionConfig{0.1, 1000.0});
    OptimizationResult result = optimizer.optimize(f, w0);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(f.gradient(result.x_opt).norm(), 1e-6);    // check gradient near zero
    EXPECT_LT(f.evaluate(result.x_opt), f.evaluate(w0)); // check objective improved
}
