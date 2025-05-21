#pragma once
#include "ObjectiveFunctionBase.hpp"
#include <Eigen/Dense>

template <int N> class ObjectiveFunction : public ObjectiveFunctionBase {
  public:
    double evaluate(const Eigen::VectorXd &x) const override {
        if (x.size() != N) {
            throw std::invalid_argument(
                "ObjectiveFunction::evaluate() called with vector of incorrect dimension");
        }
        if (!x.allFinite()) {
            throw std::invalid_argument(
                "ObjectiveFunction::evaluate() called with non-numeric vector");
        }
        return evaluateImpl(x);
    };
    Eigen::VectorXd gradient(const Eigen::VectorXd &x) const override {
        if (x.size() != N) {
            throw std::invalid_argument(
                "ObjectiveFunction::gradient() called with vector of incorrect dimension");
        }
        if (!x.allFinite()) {
            throw std::invalid_argument(
                "ObjectiveFunction::gradient() called with non-numeric vector");
        }
        Eigen::VectorXd grad = gradientImpl(x);
        if (grad.size() != N) {
            throw std::invalid_argument("gradientImpl() returned a vector of improper dimension.");
        }
        if (!grad.allFinite()) {
            throw std::invalid_argument("gradientImpl() returned a non-numeric gradient.");
        }
        return grad;
    };

    Eigen::MatrixXd hessian(const Eigen::VectorXd &x) const override {
        if (x.size() != N) {
            throw std::invalid_argument(
                "ObjectiveFunction::hessian() called with vector of incorrect dimension");
        }
        if (!x.allFinite()) {
            throw std::invalid_argument(
                "ObjectiveFunction::hessian() called with non-numeric vector");
        }
        Eigen::MatrixXd hess = hessianImpl(x);
        if (hess.rows() != N || hess.cols() != N) {
            throw std::invalid_argument("hessianImpl() returned a matrix of improper dimension.");
        }
        if (!hess.allFinite()) {
            throw std::invalid_argument("hessianImpl() returned a non-numeric matrix.");
        }
        return hess;
    }

    int sourceDimension() const override { return N; };

  protected:
    virtual double evaluateImpl(const Eigen::VectorXd &x) const {
        throw std::runtime_error("evaluateImpl() not implemented for this function.");
    };
    virtual Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const {
        throw std::runtime_error("gradientImpl() not implemented for this function.");
    };
    virtual Eigen::MaxtrixXd hessianImpl(const Eigen::VectorXd &x) const {
        throw std::runtime_error("hessianImpl() not implemented for this function.");
    };
};