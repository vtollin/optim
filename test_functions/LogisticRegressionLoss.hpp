#pragma once
#include "optim/Functions.hpp"
#include <Eigen/Dense>

class LogisticRegressionLoss : public optim::TwiceDifferentiableFunction<3> {
    Eigen::MatrixXd X_;
    Eigen::VectorXd y_;
    double lambda_;

    double evaluateImpl(const Eigen::VectorXd &w) const override {
        double loss = 0.0;
        for (int i = 0; i < X_.rows(); ++i) {
            double z = y_(i) * X_.row(i).dot(w);
            loss += std::log(1 + std::exp(-z));
        }
        loss += 0.5 * lambda_ * w.squaredNorm();
        return loss;
    }

    Eigen::VectorXd gradientImpl(const Eigen::VectorXd &w) const override {
        Eigen::VectorXd grad = lambda_ * w;
        for (int i = 0; i < X_.rows(); ++i) {
            double z = y_(i) * X_.row(i).dot(w);
            double coeff = -y_(i) / (1 + std::exp(z));
            grad += coeff * X_.row(i).transpose();
        }
        return grad;
    }

    Eigen::MatrixXd hessianImpl(const Eigen::VectorXd &w) const override {
        Eigen::MatrixXd H = lambda_ * Eigen::MatrixXd::Identity(w.size(), w.size());
        for (int i = 0; i < X_.rows(); ++i) {
            double z = y_(i) * X_.row(i).dot(w);
            double s = std::exp(z) / std::pow(1 + std::exp(z), 2);
            H += s * X_.row(i).transpose() * X_.row(i);
        }
        return H;
    }

  public:
    LogisticRegressionLoss(Eigen::MatrixXd X, Eigen::VectorXd y, double lambda)
        : X_(X), y_(y), lambda_(lambda) {}
};
