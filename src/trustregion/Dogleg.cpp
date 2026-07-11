#include "optim/trustregion/Dogleg.hpp"
#include "optim/Functions.hpp"
#include <Eigen/Dense>

using namespace optim::trustregion;

Dogleg::Dogleg(int max_iterations, optim::ConvergenceCriteria criteria, TrustRegionConfig config)
    : TrustRegionBase(max_iterations, criteria, config) {
}

SubproblemResult Dogleg::solveSubproblem(const Eigen::VectorXd &grad, const Eigen::MatrixXd &B,
                                         double delta) {
    double numerator = grad.squaredNorm();                      // g^T g
    double denominator = (grad.transpose() * B * grad).value(); // g^T B g

    Eigen::VectorXd pU = -(numerator / denominator) * grad;
    // is this just a check for positive definiteness? Could be quicker to just calculate step
    // and throw if it's non-descent. Also should probably be std::invalid_argument
    Eigen::LLT<Eigen::MatrixXd> llt(B);
    if (llt.info() == Eigen::NumericalIssue) {
        throw std::runtime_error(
            "[Dogleg] Hessian is not positive definite; dogleg requires convex problems.");
    }
    Eigen::VectorXd pB = B.llt().solve(-grad);

    Eigen::VectorXd step;
    SubproblemStatus status;

    if (pB.norm() <= delta) {
        step = pB;
        status = SubproblemStatus::INTERIOR;
    } else if (pU.norm() >= delta) {
        step = (delta / pU.norm()) * pU;
        status = SubproblemStatus::BOUNDARY;
    } else {
        Eigen::VectorXd d = pB - pU;

        double a = d.squaredNorm();
        double b = 2.0 * d.dot(pU);
        double c = pU.squaredNorm() - delta * delta;

        double discriminant = b * b - 4 * a * c;
        if (discriminant < 0) {
            throw std::runtime_error("[Dogleg] Negative discriminant in step calculation.");
        }

        double s = (-b + std::sqrt(discriminant)) / (2 * a);
        step = pU + s * d;
        status = SubproblemStatus::BOUNDARY;
    }

    return SubproblemResult{step, status};
}
