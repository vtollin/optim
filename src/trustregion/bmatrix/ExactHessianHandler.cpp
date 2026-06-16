#include "trustregion/internal/ExactHessianHandler.hpp"
#include "optim/AbstractFunctions.hpp"
#include <Eigen/Dense>
#include <stdexcept>

using optim::abstract::DifferentiableFunction;
using optim::abstract::TwiceDifferentiableFunction;

optim::trustregion::ExactHessianHandler::ExactHessianHandler() = default;

Eigen::MatrixXd optim::trustregion::ExactHessianHandler::initialize(
    const DifferentiableFunction &f, const Eigen::VectorXd &x) {
    try {
        return dynamic_cast<const TwiceDifferentiableFunction &>(f).hessian(x);
    } catch (const std::bad_cast &) {
        throw std::invalid_argument(
            "[ExactHessianHandler] initialize() requires a TwiceDifferentiableFunction.");
    }
}

Eigen::MatrixXd optim::trustregion::ExactHessianHandler::getB(
    const DifferentiableFunction &f, const Eigen::VectorXd &x) {
    try {
        return dynamic_cast<const TwiceDifferentiableFunction &>(f).hessian(x);
    } catch (const std::bad_cast &) {
        throw std::invalid_argument(
            "[ExactHessianHandler] getB() requires a TwiceDifferentiableFunction.");
    }
}
