#include "GeneralizedRosenbrock.hpp"
#include "optim/linesearch/BFGS.hpp"
#include "optim/linesearch/Newton.hpp"
#include "optim/linesearch/SteepestDescent.hpp"
#include "optim/trustregion/Dogleg.hpp"
#include "optim/trustregion/MoreSorensen.hpp"
#include "optim/trustregion/SteihaugCG.hpp"
#include <Eigen/Dense>
#include <benchmark/benchmark.h>

using namespace optim;

Eigen::VectorXd generalized_rosenbrock_start(int n) {
    Eigen::VectorXd x0(n);
    for (int i = 0; i < n; ++i) {
        x0(i) = (i % 2 == 0) ? -1.2 : 1.0;
    }
    return x0;
}

template <typename Optimizer> static void BM_Rosenbrock(benchmark::State &state) {
    const int n = state.range(0);
    GeneralizedRosenbrock f(n);
    Eigen::VectorXd x0 = generalized_rosenbrock_start(n);
    Optimizer optimizer;
    double total_iters = 0;
    for (auto _ : state) {
        auto result = optimizer.optimize(f, x0);
        total_iters += result.iterations;
        benchmark::DoNotOptimize(result);
    }
    state.counters["iters"] = total_iters / state.iterations();
}

BENCHMARK_TEMPLATE(BM_Rosenbrock, trustregion::MoreSorensen)
    ->Arg(2)
    ->Arg(50)
    ->Arg(200)
    ->Name("BM_MoreSorensen_Rosenbrock");
BENCHMARK_TEMPLATE(BM_Rosenbrock, trustregion::SteihaugCG)
    ->Arg(2)
    ->Arg(50)
    ->Arg(200)
    ->Name("BM_SteihaugCG_Rosenbrock");
BENCHMARK_TEMPLATE(BM_Rosenbrock, trustregion::Dogleg)
    ->Arg(2)
    ->Arg(50)
    ->Arg(200)
    ->Name("BM_Dogleg_Rosenbrock");
BENCHMARK_TEMPLATE(BM_Rosenbrock, linesearch::SteepestDescent)
    ->Arg(2)
    ->Arg(50)
    ->Arg(200)
    ->Name("BM_SteepestDescent_Rosenbrock");
BENCHMARK_TEMPLATE(BM_Rosenbrock, linesearch::BFGS)
    ->Arg(2)
    ->Arg(50)
    ->Arg(200)
    ->Name("BM_BFGS_Rosenbrock");
BENCHMARK_TEMPLATE(BM_Rosenbrock, linesearch::Newton)
    ->Arg(2)
    ->Arg(50)
    ->Arg(200)
    ->Name("BM_Newton_Rosenbrock");