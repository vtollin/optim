![CI](https://github.com/vtollin/optimization/actions/workflows/ci.yml/badge.svg)

# optim

A C++ library implementing line search and trust region methods for unconstrained optimization, built on a shared `Function` / `OptimizationResult` interface. Correctness is verified rather than assumed: analytical gradients and Hessians are checked against central finite differences, and every solver is tested for convergence on standard nonconvex benchmarks (Rosenbrock, Himmelblau, Wood).

## Features

- Line search: Steepest Descent, BFGS (with Powell damping), Newton (with modified Cholesky)
- Trust region: Dogleg, More–Sorensen, Steihaug-CG
- Pluggable step-length policies (Armijo backtracking, Strong Wolfe) for line search
- Observer hooks (`IterationObserver`, `BFGSObserver`, `NewtonObserver`) for inspecting iteration
  history, Hessian modifications, and BFGS damping events without touching the core loop
- Finite-difference gradient/Hessian checkers (`optim::diagnostics`) for validating a new `Function`
- Unit tests for every shared base class and algorithm, plus integration tests against standard
  benchmark problems

## Quickstart

```cpp
#include "optim/Functions.hpp"
#include "optim/trustregion/MoreSorensen.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

// Rosenbrock's function: a classic ill-conditioned, nonconvex benchmark with a curved
// valley leading to the global minimum at (1, 1).
class Rosenbrock : public optim::TwiceDifferentiableFunction {
  public:
    Rosenbrock() : optim::TwiceDifferentiableFunction(2) {}

  protected:
    double evaluateImpl(const Eigen::VectorXd &x) const override {
        return std::pow(1.0 - x(0), 2) + 100.0 * std::pow(x(1) - x(0) * x(0), 2);
    }
    Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const override {
        Eigen::VectorXd grad(2);
        grad(0) = -2.0 * (1.0 - x(0)) - 400.0 * x(0) * (x(1) - x(0) * x(0));
        grad(1) = 200.0 * (x(1) - x(0) * x(0));
        return grad;
    }
    Eigen::MatrixXd hessianImpl(const Eigen::VectorXd &x) const override {
        Eigen::Matrix2d H;
        H(0, 0) = 2.0 - 400.0 * x(1) + 1200.0 * x(0) * x(0);
        H(0, 1) = H(1, 0) = -400.0 * x(0);
        H(1, 1) = 200.0;
        return H;
    }
};

int main() {
    Rosenbrock f;
    Eigen::VectorXd x0(2);
    x0 << -1.2, 1.0;

    optim::trustregion::MoreSorensen optimizer;
    optim::OptimizationResult result = optimizer.optimize(f, x0);

    std::cout << "x*  = " << result.x_opt.transpose() << "\n"
              << "f*  = " << result.f_val << "\n"
              << "iters = " << result.iterations << "\n"
              << "converged = " << std::boolalpha << result.converged << "\n";
}
```

## Design Space: Line Search vs. Trust Region

Both families solve the same problem (minimize a smooth `f`) but differ in how they leverage
their local quadratic model and guarantee progress. Line search picks a direction first, and then
finds a step along it that satisfies a sufficient-decrease condition. Trust region
bounds the step by a radius and grows or shrinks that radius based on how well the quadratic model
predicted the actual decrease.

| Method | Family | Cost / iteration | When to use |
|---|---|---|---|
| Steepest Descent | Line search | O(n): gradient evaluations | Cheap baseline; Hessian unavailable or expensive; only competitive on well-scaled problems (slow on ill-conditioned ones) |
| BFGS | Line search | O(n²): rank-2 update + matrix-vector products | Default choice for smooth unconstrained problems when the exact Hessian is unavailable or costly; scales well to small–medium n |
| Newton (line search) | Line search | O(n³): modified Cholesky of the exact Hessian | Exact Hessian is cheap to form and n is small–medium; fastest local convergence near the solution |
| Dogleg | Trust region | O(n³): one factorization/solve for the Newton step, then a closed-form path | Cheap approximate trust-region step when the exact Hessian is affordable to form once per iteration |
| More–Sorensen | Trust region | O(n³) per inner iteration, several inner iterations per step | Use when an indefinite Hessian requires an accurate subproblem solve and repeated factorizations are affordable. |
| Steihaug-CG | Trust region | O(n²) per CG iteration: matrix-vector products only, no factorization | Large problems where factoring the Hessian is too expensive |

## Build & Test

Requires CMake ≥ 3.20, a C++17 compiler, and Eigen3.

```bash
# macOS
brew install eigen

# Linux
sudo apt-get install libeigen3-dev
```

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

GoogleTest is fetched automatically via CMake's `FetchContent`, so no separate install step is needed. The first `cmake` invocation requires network access. 

## Usage

Add `optim` to your project via CMake's `FetchContent`: 

```cmake
include(FetchContent)
FetchContent_Declare(
  optim
  GIT_REPOSITORY https://github.com/vtollin/optim
  GIT_TAG main
)
FetchContent_MakeAvailable(optim)

target_link_libraries(your_target PRIVATE optim)
```

## Architecture

### Class Hierarchy
- `Function → DifferentiableFunction → TwiceDifferentiableFunction`
- `OptimizerBase → LineSearchBase<FuncType> / TrustRegionBase`
- The `Function` hierarchy ensures that the interfaces are segregated so that users only need to implement the methods they need.
`Function` owns `evaluate()`, `DifferentiableFunction` adds `gradient()`, and `TwiceDifferentiableFunction` adds `hessian()`. 
- `OptimizerBase` owns `max_iterations` and `ConvergenceCriteria`, a struct of convergence tolerances (see [Convergence Criteria](#convergence-criteria)). Both are shared across optimizer families. 
- `LineSearchBase` is templated on `FuncType` since the subclass requirements vary: `SteepestDescent` only uses gradients while `Newton` and `BFGS` require Hessians. In contrast, all trust region optimizers require a `TwiceDifferentiableFunction` to form the
quadratic model, so `TrustRegionBase` is not templated. 
- `LineSearchBase` holds a public `optimize()` that runs the driver loop and calls pure-virtual `computeDirection()`, which is overridden by subclasses. Similarly, `TrustRegionBase` holds a public `optimize()` that calls pure-virtual `solveSubproblem()`. This Template Method pattern ensures that the common driver loop in each family is only written once. Both base classes also expose a public `addObserver()` for iteration tracing (see [Observability](#observability)). 

### Validation at the Boundary
- The `Function` hierarchy is designed according to the Non-Virtual Interface pattern: public methods (`evaluate()`, `gradient()`, `hessian()`) are non-virtual and delegate to private virtual helpers (`evaluateImpl()`, `gradientImpl()`, `hessianImpl()`). This design allows dimension and NaN guards to live in the public wrapper.
- Because the guards live in the non-virtual wrapper, they can't be skipped or forgotten in a new `Function` implementation. Validation happens once, at the `Function` boundary, rather than being reimplemented in every optimizer. `Dogleg`, `Newton`, `BFGS`, and every other algorithm can call `evaluate()`, `gradient()`, or `hessian()` and trust the result. No per-algorithm bounds or NaN checks are needed. 

### Observability
- `IterationObserver` (separate line search and trust region variants).
-  Algorithm-specific observers: `BFGSObserver` and `NewtonObserver`
- Each `IterationObserver` exposes a pure-virtual `onIteration()` and virtual (default no-ops) `onStart()` and `onFinish()` methods. Users can override `IterationObserver` for their use case (plotting, formatted printing, etc). `BFGSObserver` and `NewtonObserver` similarly hold pure-virtual `onBFGSupdate()` and `onModifiedCholesky()` respectively for algorithm-specific tracing.
- `IterationObserver` is not shared across line search and trust region families because each needs a distinct `IterationInfo` passed to `onIteration()`: trust region's replaces `StepStatus` with `SubproblemStatus` and adds `rho`, `delta`, and `accepted`. To avoid a bloated struct with optional fields, a shared `IterationObserver` would need to be templated on `IterationInfo`, which would also require a templated manager class to hold an optimizer's observers. This added complexity does not earn its keep against ~15 lines of duplicated, unchanging logic that will not need to be copied again. 

### Convergence Criteria
- `ConvergenceCriteria { grad_tol, step_tol, f_tol }`, shared across every algorithm.
-  `grad_tol` has a default value of `1e-8`, while `step_tol` and `f_tol` are optional. This distinction stems from what each actually detects: `grad_tol` is the convergence criterion, while `step_tol`/`f_tol` are stall detectors. The stall detectors are used to detect when the step or function value has stopped changing meaningfully across iterations. This is an optional addition to reduce unnecessary computations on a rare failure mode or when the user only needs a certain level of precision. Otherwise, optimizer behavior is simple and predictable: it grinds towards convergence until `max_iterations` is reached. 

### Testing Strategy
- `unit/functions` validates each benchmark function's analytical gradient and Hessian against central finite-difference approximations. This ensures that the optimizers are validated against demonstrably correct test fixtures. 
- `unit/base` tests the shared driver loops in isolation. `LineSearchBase` and `TrustRegionBase` are abstract, so they must be tested through complete subclasses. `LineSearchBase` is tested through `SteepestDescent` since its `computeDirection()` is trivial (it simply negates the gradient). `TrustRegionBase` is tested through
a custom test-only subclass, since none of its subclasses are trivial enough to isolate its behavior. 
- `unit/steplengthpolicy` tests the step-length algorithms (`ArmijoBacktracking` and `StrongWolfe`) separately from the direction-computing line search classes that use them. This verifies that the step-length logic is correct in isolation. 
- `integration/{linesearch,trustregion}` validates the full algorithms against benchmark problems. 

## Validation

Every algorithm converges to a known minimizer from a standard textbook starting point on three
benchmarks. Numbers below are real, measured runs (default `grad_tol = 1e-8`), not asserted bounds. Values are rounded to three significant figures. 

### Rosenbrock
Global minimum `f(1, 1) = 0`, starting point `x₀ = (-1.2, 1.0)`. Every method converges to
`x* = (1, 1)`.

| Method | Iterations | \|f - f\*\| |
|---|---|---|
| Steepest Descent | 17097 | 7.34e-17 |
| BFGS | 33 | 1.14e-20 |
| Newton | 75 | 1.30e-19 |
| Dogleg | 29 | 0 (exact) |
| More–Sorensen | 27 | 1.14e-20 |
| Steihaug-CG | 38 | 1.23e-30 |

Dogleg's final iterate coincided bit-for-bit with the exact double representation of the minimizer, so f* evaluates to exactly 0.0. Other optimizers converged to iterates within a few ULPs of (1,1) rather than exactly on it, yielding the small nonzero residuals shown above.  

### Himmelblau
Four symmetric global minima, all with `f = 0`, starting point `x₀ = (-0.27, -0.923)`.

| Method | Iterations |  \|f - f\*\| | x* |
|---|---|---|---|
| Steepest Descent | 289 | 6.80e-19 | (3.584, -1.848) |
| BFGS | 12 | 1.55e-20 | (3.584, -1.848) |
| Newton | 16 | 7.89e-31| (3, 2) |
| Dogleg | 10 | 7.89e-31 | (3.584, -1.848) |
| More–Sorensen | 10 | 1.54e-25 | (3.584, -1.848) |
| Steihaug-CG | 12 | 3.99e-27 | (3.584, -1.848) |

Newton converges to `(3, 2)` while every other method converges to 
`(3.584, -1.848)`. Both are exact, valid minima, just reached via different paths.

### Wood
Global minimum `f(1, 1, 1, 1) = 0`, starting point `x₀ = (-3, -1, -3, -1)`. Every method converges
to `x* = (1, 1, 1, 1)`.

| Method | Iterations | \|f - f\*\|  |
|---|---|---|
| Steepest Descent | 11248 | 6.21e-17 |
| BFGS | 35 | 4.40e-22 |
| Newton | 215 | 3.08e-24 |
| Dogleg | 3922 | 2.60e-21 |
| More–Sorensen | 44 | 3.62e-25 |
| Steihaug-CG | 112 | 1.80e-23 |

Steepest Descent makes the design-space table concrete: on the ill-conditioned Rosenbrock/Wood
valleys it needs 300–500x more iterations than BFGS or Newton to reach the same tolerance.

## References

The algorithms here follow Nocedal & Wright, *Numerical Optimization* (2nd ed.) closely: the
dogleg method and More–Sorensen's secular-equation solve (Ch. 4), Steihaug-CG (Ch. 7), Powell-damped
BFGS (Ch. 6), and modified-Cholesky Newton (Ch. 3).

## License

[MIT](LICENSE)
