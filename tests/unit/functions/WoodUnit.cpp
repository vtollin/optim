#include "Wood.hpp"
#include "optim/diagnostics/GradientChecker.hpp"
#include "optim/diagnostics/HessianChecker.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>

using optim::diagnostics::gradient_check;
using optim::diagnostics::hessian_check;

struct Pt4 {
    double x0, x1, x2, x3;
};

class WoodTest : public ::testing::TestWithParam<Pt4> {
  protected:
    Wood f;
};

TEST_P(WoodTest, GradientCheck) {
    Eigen::VectorXd x(4);
    x << GetParam().x0, GetParam().x1, GetParam().x2, GetParam().x3;
    auto r = gradient_check(f, x);
    EXPECT_TRUE(r.result) << "max_abs_error=" << r.max_abs_error
                          << " max_rel_error=" << r.max_rel_error
                          << " worst_coord=" << r.worst_coord;
}

TEST_P(WoodTest, HessianCheck) {
    Eigen::VectorXd x(4);
    x << GetParam().x0, GetParam().x1, GetParam().x2, GetParam().x3;
    auto r = hessian_check(f, x);
    EXPECT_TRUE(r.result) << "max_abs_error=" << r.max_abs_error
                          << " max_rel_error=" << r.max_rel_error << " worst=(" << r.worst_row
                          << "," << r.worst_col << ")";
    EXPECT_TRUE(r.analytic_symmetric);
}

INSTANTIATE_TEST_SUITE_P(Points, WoodTest,
                         ::testing::Values(Pt4{1.0, 1.0, 1.0, 1.0},     // global minimum
                                           Pt4{-3.0, -1.0, -3.0, -1.0}, // suggested starting point
                                           Pt4{0.5, 0.5, 0.5, 0.5},     // midpoint toward minimum
                                           Pt4{2.0, 1.0, 0.5, -0.5} // asymmetric off-minimum point
                                           ));
