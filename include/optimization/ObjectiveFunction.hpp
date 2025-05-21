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
            throw std::invalid_argument("GradientImpl() returned a vector of improper dimension.");
        }
        if (!grad.allFinite()) {
            throw std::invalid_argument("GradientImpl() returned a non-numeric gradient.");
        }
        return grad;
    };

    int sourceDimension() const override { return N; };

  protected:
    virtual double evaluateImpl(const Eigen::VectorXd &x) const = 0;
    virtual Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const = 0;
};