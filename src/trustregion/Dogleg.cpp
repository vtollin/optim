#include "optim/trustregion/Dogleg.hpp"
#include "optim/logger/Logger.hpp"
#include "optim/Functions.hpp"
#include <Eigen/Dense>
#include <string>

using namespace optim::trustregion;
using optim::OptimizationResult;
using optim::TwiceDifferentiableFunction;

Dogleg::Dogleg(int max_iterations, optim::ConvergenceCriteria criteria, double delta_init,
               double delta_max, double eta, std::shared_ptr<optim::logger::Logger> logger)
    : TrustRegionBase(max_iterations, criteria, delta_init, delta_max, eta, logger) {
}

OptimizationResult Dogleg::optimize(const TwiceDifferentiableFunction &f,
                                    const Eigen::VectorXd &x0) {
    int k = 0;
    bool converged = false;
    std::string msg = "Failed to converge";

    Eigen::VectorXd x = x0;

    int n = f.sourceDimension();
    if (n != x.size()) {
        throw std::invalid_argument(
            "[Dogleg] Initial vector is not in the source of objective function.");
    }
    if (delta_ < 0.0) {
        delta_ = std::min(1.0, f.gradient(x0).norm());
    }

    for (; k < max_iterations_; ++k) {
        Eigen::VectorXd grad = f.gradient(x);
        if (grad.norm() < criteria_.grad_tol) {
            converged = true;
            msg = "Converged: gradient fell below tolerance.";
            break;
        }
        QuadraticModel m;
        m.f_x = f.evaluate(x);
        m.g = f.gradient(x);
        m.B = f.hessian(x);

        double numerator = m.g.squaredNorm();                       // g^T g
        double denominator = (m.g.transpose() * m.B * m.g).value(); // g^T B g

        Eigen::VectorXd pU = -(numerator / denominator) * m.g;
        Eigen::LLT<Eigen::MatrixXd> llt(m.B);
        if (llt.info() == Eigen::NumericalIssue) {
            throw std::runtime_error(
                "[Dogleg] Hessian is not positive definite; dogleg requires convex problems.");
        }
        Eigen::VectorXd pB = m.B.llt().solve(-m.g);

        Eigen::VectorXd step;

        if (pB.norm() <= delta_) {
            step = pB;
        } else if (pU.norm() >= delta_) {
            step = (delta_ / pU.norm()) * pU;
        } else {
            Eigen::VectorXd d = pB - pU;

            double a = d.squaredNorm();
            double b = 2.0 * d.dot(pU);
            double c = pU.squaredNorm() - delta_ * delta_;

            double discriminant = b * b - 4 * a * c;
            if (discriminant < 0) {
                throw std::runtime_error("[Dogleg] Negative discriminant in step calculation.");
            }

            double s = (-b + std::sqrt(discriminant)) / (2 * a);
            step = pU + s * d;
        }

        if (step.norm() < criteria_.step_tol * (x.norm() + 1.0)) {
            converged = true;
            msg = "Converged: Step size fell below tolerance.";
            x += step;
            break;
        }
        double f1 = f.evaluate(x + step);
        if (std::abs(f1 - m.f_x) < criteria_.f_tol * (std::abs(m.f_x) + 1.0)) {
            converged = true;
            msg = "Converged: Objective function change fell below tolerance.";
            x += step;
            break;
        }
        UpdateResult result = update(f, m, x, step);
        if (logger_ && logger_->getVerbosity() == optim::logger::Verbosity::INFO) {
            logger_->logIteration(
                {k, x, grad, step, m.f_x, result.rho, result.accepted, result.delta_old});
        }
        if (result.accepted) {
            x = x + step;
        }
    }
    return OptimizationResult{x, f.evaluate(x), k, converged, msg};
}
