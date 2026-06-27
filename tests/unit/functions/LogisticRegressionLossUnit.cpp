#include "LogisticRegressionLoss.hpp"
#include "optim/diagnostics/GradientChecker.hpp"
#include "optim/diagnostics/HessianChecker.hpp"
#include <Eigen/Dense>
#include <array>
#include <gtest/gtest.h>

using optim::diagnostics::gradient_check;
using optim::diagnostics::hessian_check;

// 10 samples, 3 features; same dataset used in DoglegIntegration.LogisticRegression
static Eigen::MatrixXd makeX() {
    Eigen::MatrixXd X(10, 3);
    X << 1, 2, 3, 2, 1, 3, 3, 4, 1, 4, 3, 2, 1, 0, 2, 0, 1, 1, 3, 3, 3, 2, 2, 2, 1, 4, 3, 4, 1, 0;
    return X;
}

static Eigen::VectorXd makeY() {
    Eigen::VectorXd y(10);
    y << 1, -1, 1, -1, 1, -1, 1, -1, 1, -1;
    return y;
}

class LogisticRegressionLossTest : public ::testing::TestWithParam<std::array<double, 3>> {
  protected:
    LogisticRegressionLoss f{makeX(), makeY(), 0.1};
};

TEST_P(LogisticRegressionLossTest, GradientCheck) {
    auto w_arr = GetParam();
    Eigen::VectorXd w(3);
    w << w_arr[0], w_arr[1], w_arr[2];
    auto r = gradient_check(f, w);
    EXPECT_TRUE(r.result) << "max_abs_error=" << r.max_abs_error
                          << " max_rel_error=" << r.max_rel_error
                          << " worst_coord=" << r.worst_coord;
}

TEST_P(LogisticRegressionLossTest, HessianCheck) {
    auto w_arr = GetParam();
    Eigen::VectorXd w(3);
    w << w_arr[0], w_arr[1], w_arr[2];
    auto r = hessian_check(f, w);
    EXPECT_TRUE(r.result) << "max_abs_error=" << r.max_abs_error
                          << " max_rel_error=" << r.max_rel_error << " worst=(" << r.worst_row
                          << "," << r.worst_col << ")";
    EXPECT_TRUE(r.analytic_symmetric);
}

INSTANTIATE_TEST_SUITE_P(
    Weights, LogisticRegressionLossTest,
    ::testing::Values(std::array<double, 3>{0.0, 0.0, 0.0},  // zero init: symmetric gradient
                      std::array<double, 3>{1.0, -1.0, 0.5}, // generic non-zero weights
                      std::array<double, 3>{0.5, 0.5, -0.5}, // mixed-sign weights
                      std::array<double, 3>{3.0, -3.0, 1.0}
                      // large weights; stresses Hessian numerics
                      ));
