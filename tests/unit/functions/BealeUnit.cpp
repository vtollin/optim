#include "Beale.hpp"
#include "optim/diagnostics/GradientChecker.hpp"
#include "optim/diagnostics/HessianChecker.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>

using optim::diagnostics::gradient_check;
using optim::diagnostics::hessian_check;

struct Pt2 {
    double x0, x1;
};

class BealeTest : public ::testing::TestWithParam<Pt2> {
  protected:
    Beale f;
};

TEST_P(BealeTest, GradientCheck) {
    Eigen::VectorXd x(2);
    x << GetParam().x0, GetParam().x1;
    auto r = gradient_check(f, x);
    EXPECT_TRUE(r.result) << "max_abs_error=" << r.max_abs_error
                          << " max_rel_error=" << r.max_rel_error
                          << " worst_coord=" << r.worst_coord;
}

TEST_P(BealeTest, HessianCheck) {
    Eigen::VectorXd x(2);
    x << GetParam().x0, GetParam().x1;
    auto r = hessian_check(f, x);
    EXPECT_TRUE(r.result) << "max_abs_error=" << r.max_abs_error
                          << " max_rel_error=" << r.max_rel_error << " worst=(" << r.worst_row
                          << "," << r.worst_col << ")";
    EXPECT_TRUE(r.analytic_symmetric);
}

INSTANTIATE_TEST_SUITE_P(Points, BealeTest,
                         ::testing::Values(Pt2{3.0, 0.5},  // global minimum
                                           Pt2{1.0, 1.0},  // suggested starting point
                                           Pt2{0.0, 0.0},  // origin
                                           Pt2{-1.0, 0.5}, // negative x0 region
                                           Pt2{-1.0, -2.0} // negative x0 and x1 region
                                           ));
