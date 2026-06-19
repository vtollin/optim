// src/trustregion/internal/BFGSHandler.cpp
#include "trustregion/internal/BFGSHandler.hpp"
#include "optim/Functions.hpp"
#include <stdexcept>

using optim::DifferentiableFunction;

optim::trustregion::BFGSHandler::BFGSHandler() : x_prev_(), grad_prev_(), B_prev_() {
}

Eigen::MatrixXd optim::trustregion::BFGSHandler::initialize(const DifferentiableFunction &f,
                                                             const Eigen::VectorXd &x) {
    x_prev_ = x;
    grad_prev_ = f.gradient(x);
    int n = f.sourceDimension();
    B_prev_ = Eigen::MatrixXd::Identity(n, n);
    return B_prev_;
}

Eigen::MatrixXd optim::trustregion::BFGSHandler::getB(const DifferentiableFunction &f,
                                                       const Eigen::VectorXd &x) {
    if (x_prev_.size() == 0 || grad_prev_.size() == 0 || B_prev_.size() == 0) {
        throw std::runtime_error("[BFGSHandler]: Called getB() before initialize()");
    }

    Eigen::VectorXd grad = f.gradient(x);
    Eigen::VectorXd s = x - x_prev_;
    Eigen::VectorXd y = grad - grad_prev_;

    Eigen::VectorXd Bs = B_prev_ * s;
    double ys = y.dot(s);
    double sBs = s.dot(Bs);

    Eigen::VectorXd ybar;
    if (ys >= 0.2 * sBs) {
        ybar = y;
    } else {
        double theta = (0.8 * sBs) / (sBs - ys);
        ybar = theta * y + (1.0 - theta) * Bs;
    }

    double denom1 = s.dot(ybar); // = s^T ybar > 0
    double denom2 = sBs;         // = s^T B s > 0

    Eigen::VectorXd u = std::sqrt(1.0 / denom1) * ybar;
    Eigen::VectorXd v = std::sqrt(1.0 / denom2) * Bs;

    Eigen::MatrixXd B = B_prev_;
    B.noalias() += u * u.transpose();
    B.noalias() -= v * v.transpose();

    x_prev_ = x;
    grad_prev_ = grad;
    B_prev_ = B;

    return B;
}
