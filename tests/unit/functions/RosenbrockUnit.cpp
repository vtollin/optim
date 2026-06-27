#include "Rosenbrock.hpp"
#include "optim/diagnostics/GradientChecker.hpp"
#include "optim/diagnostics/HessianChecker.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>

using optim::diagnostics::gradient_check;
using optim::diagnostics::hessian_check;

struct Pt2 {
    double x0, x1;
};

class RosenbrockTest : public ::testing::TestWithParam<Pt2> {
  protected:
    Rosenbrock f;
};

TEST_P(RosenbrockTest, GradientCheck) {
    Eigen::VectorXd x(2);
    x << GetParam().x0, GetParam().x1;
    auto r = gradient_check(f, x);
    EXPECT_TRUE(r.result) << "max_abs_error=" << r.max_abs_error
                          << " max_rel_error=" << r.max_rel_error
                          << " worst_coord=" << r.worst_coord;
}

TEST_P(RosenbrockTest, HessianCheck) {
    Eigen::VectorXd x(2);
    x << GetParam().x0, GetParam().x1;
    auto r = hessian_check(f, x);
    EXPECT_TRUE(r.result) << "max_abs_error=" << r.max_abs_error
                          << " max_rel_error=" << r.max_rel_error << " worst=(" << r.worst_row
                          << "," << r.worst_col << ")";
    EXPECT_TRUE(r.analytic_symmetric);
}

INSTANTIATE_TEST_SUITE_P(Points, RosenbrockTest,
                         ::testing::Values(Pt2{1.0, 1.0},  // global minimum
                                           Pt2{-1.2, 1.0}, // classic starting point
                                           Pt2{0.5, 0.25}, // on the banana curve y=x^2
                                           Pt2{2.0, 4.0}   // past minimum, on the valley
                                           ));
