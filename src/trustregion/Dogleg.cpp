#include "optim/trustregion/Dogleg.hpp"
#include "optim/Functions.hpp"
#include <Eigen/Dense>

using namespace optim::trustregion;

Dogleg::Dogleg(int max_iterations, optim::ConvergenceCriteria criteria, TrustRegionConfig config)
    : TrustRegionBase(max_iterations, criteria, config) {
}

SubproblemResult Dogleg::solveSubproblem(const Eigen::VectorXd &grad, const Eigen::MatrixXd &B,
                                         double delta) {
    const double grad_norm = grad.norm();
    const double d2 = delta * delta;

    Eigen::LLT<Eigen::MatrixXd> llt(B);

    if (llt.info() != Eigen::Success) { // not PD
        const double gBg = grad.dot(B * grad);
        Eigen::VectorXd step = -(delta / grad_norm) * grad;
        if (gBg > 0) {
            double tau = std::min(1.0, grad_norm * grad.squaredNorm() / (delta * gBg));
            step *= tau;
        }
        return SubproblemResult{step, SubproblemStatus::NEGATIVECURVATURE};
    }

    Eigen::VectorXd pB = llt.solve(-grad);
    if (pB.squaredNorm() <= d2) {
        return SubproblemResult{pB, SubproblemStatus::INTERIOR};
    }

    const double gBg = grad.dot(B * grad);
    Eigen::VectorXd pU = -(grad.squaredNorm() / gBg) * grad;
    const double pU_norm = pU.norm();

    if (pU_norm >= delta) {
        return SubproblemResult{(delta / pU_norm) * pU, SubproblemStatus::BOUNDARY};
    }

    Eigen::VectorXd d = pB - pU;
    double a = d.squaredNorm();
    double b = 2.0 * d.dot(pU);
    double c = pU.squaredNorm() - d2;
    double s = (-b + std::sqrt(b * b - 4.0 * a * c)) / (2.0 * a);
    return SubproblemResult{pU + s * d, SubproblemStatus::BOUNDARY};
}
