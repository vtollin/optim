#include "GeneralizedRosenbrock.hpp"
#include "optim/diagnostics/GradientChecker.hpp"
#include "optim/diagnostics/HessianChecker.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <vector>

using optim::diagnostics::gradient_check;
using optim::diagnostics::hessian_check;

struct PtN {
    std::vector<double> x;
};

namespace {
Eigen::VectorXd toVector(const std::vector<double> &x) {
    Eigen::VectorXd v(static_cast<int>(x.size()));
    for (std::size_t i = 0; i < x.size(); ++i) v(static_cast<int>(i)) = x[i];
    return v;
}
} // namespace

class GeneralizedRosenbrockTest : public ::testing::TestWithParam<PtN> {};

TEST_P(GeneralizedRosenbrockTest, GradientCheck) {
    Eigen::VectorXd x = toVector(GetParam().x);
    GeneralizedRosenbrock f(static_cast<int>(x.size()));
    auto r = gradient_check(f, x);
    EXPECT_TRUE(r.result) << "max_abs_error=" << r.max_abs_error
                          << " max_rel_error=" << r.max_rel_error
                          << " worst_coord=" << r.worst_coord;
}

TEST_P(GeneralizedRosenbrockTest, HessianCheck) {
    Eigen::VectorXd x = toVector(GetParam().x);
    GeneralizedRosenbrock f(static_cast<int>(x.size()));
    auto r = hessian_check(f, x);
    EXPECT_TRUE(r.result) << "max_abs_error=" << r.max_abs_error
                          << " max_rel_error=" << r.max_rel_error << " worst=(" << r.worst_row
                          << "," << r.worst_col << ")";
    EXPECT_TRUE(r.analytic_symmetric);
}

INSTANTIATE_TEST_SUITE_P(Points, GeneralizedRosenbrockTest,
                         ::testing::Values(
                             PtN{{1.0, 1.0}},                          // n=2 global minimum
                             PtN{{-1.2, 1.0}},                         // n=2 classic starting point
                             PtN{{1.0, 1.0, 1.0}},                     // n=3 global minimum
                             PtN{{-1.2, 1.0, -1.2}},                   // n=3 off minimum
                             PtN{{1.0, 1.0, 1.0, 1.0, 1.0}},           // n=5 global minimum
                             PtN{{-1.2, 1.0, -1.2, 1.0, -1.2}},        // n=5 alternating start
                             PtN{{0.5, 0.25, 1.5, -0.5, 2.0, 0.0}}     // n=6 generic interior point
                             ));
