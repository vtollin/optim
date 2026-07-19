#include "optim/trustregion/SteihaugCG.hpp"
#include <Eigen/Dense>
#include <limits>

using namespace optim::trustregion;

SteihaugCG::SteihaugCG(int max_iterations, optim::ConvergenceCriteria criteria,
                       TrustRegionConfig config)
    : TrustRegionBase(max_iterations, criteria, config) {};

SubproblemResult SteihaugCG::solveSubproblem(const Eigen::VectorXd &grad, const Eigen::MatrixXd &B,
                                             double delta) {
    Eigen::Index n = grad.size();
    // using standard eta for superlinear convergence
    double tol = std::min(0.5, std::sqrt(grad.norm())) * grad.norm();
    Eigen::VectorXd z = Eigen::VectorXd::Zero(n); // CG iterate
    Eigen::VectorXd r = grad;                     // CG residual
    Eigen::VectorXd d = -r;                       // CG direction
    double iters = std::max(n, Eigen::Index(5));
    for (Eigen::Index j = 0; j < iters; ++j) {
        Eigen::VectorXd Bd = B * d;
        double dBd = d.dot(Bd); // Curvature along d
        if (dBd <= 0) {
            double tau = computeBoundaryTau(z, d, delta);
            return SubproblemResult{z + tau * d, SubproblemStatus::BOUNDARY};
        }
        double alpha = r.squaredNorm() / dBd;
        Eigen::VectorXd z_next = z + alpha * d;
        if (z_next.norm() >= delta) {
            double tau = computeBoundaryTau(z, d, delta);
            return SubproblemResult{z + tau * d, SubproblemStatus::BOUNDARY};
        }
        Eigen::VectorXd r_next = r + alpha * Bd;
        if (r_next.norm() < tol) {
            return SubproblemResult{z_next, SubproblemStatus::INTERIOR};
        }
        double beta = r_next.squaredNorm() / r.squaredNorm();
        d = -r_next + beta * d;
        r = r_next;
        z = z_next;
    }
    return SubproblemResult{z, SubproblemStatus::INTERIOR};
}

double SteihaugCG::computeBoundaryTau(const Eigen::VectorXd &z, const Eigen::VectorXd &d,
                                      double delta) {
    double a = d.squaredNorm();
    double b = 2.0 * z.dot(d);
    double c = z.squaredNorm() - delta * delta;
    double q = -0.5 * (b + std::copysign(std::sqrt(b * b - 4 * a * c), b));
    double tau1 = q / a;
    double tau2 = c / q;
    return std::max(tau1, tau2);
}