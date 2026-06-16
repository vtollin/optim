#pragma once
#include "AbstractFunctions.hpp"
#include <Eigen/Dense>
#include <stdexcept>

namespace optim {

template <int N> class Function : public abstract::Function {
  public:
    double evaluate(const Eigen::VectorXd &x) const final {
        if (x.size() != N) {
            throw std::invalid_argument(
                "Function::evaluate() called with vector of incorrect dimension");
        }
        if (!x.allFinite()) {
            throw std::invalid_argument("Function::evaluate() called with non-numeric vector");
        }
        return evaluateImpl(x);
    }

    int sourceDimension() const final { return N; }

  protected:
    virtual double evaluateImpl(const Eigen::VectorXd &x) const = 0;
};

template <int N> class DifferentiableFunction : public abstract::DifferentiableFunction {
  public:
    double evaluate(const Eigen::VectorXd &x) const final {
        if (x.size() != N) {
            throw std::invalid_argument(
                "DifferentiableFunction::evaluate() called with vector of incorrect dimension");
        }
        if (!x.allFinite()) {
            throw std::invalid_argument(
                "DifferentiableFunction::evaluate() called with non-numeric vector");
        }
        return evaluateImpl(x);
    }

    int sourceDimension() const final { return N; }

    Eigen::VectorXd gradient(const Eigen::VectorXd &x) const final {
        if (x.size() != N) {
            throw std::invalid_argument(
                "DifferentiableFunction::gradient() called with vector of incorrect dimension");
        }
        if (!x.allFinite()) {
            throw std::invalid_argument(
                "DifferentiableFunction::gradient() called with non-numeric vector");
        }
        Eigen::VectorXd grad = gradientImpl(x);
        if (grad.size() != N) {
            throw std::invalid_argument("gradientImpl() returned a vector of improper dimension.");
        }
        if (!grad.allFinite()) {
            throw std::invalid_argument("gradientImpl() returned a non-numeric gradient.");
        }
        return grad;
    }

  protected:
    virtual double evaluateImpl(const Eigen::VectorXd &x) const = 0;
    virtual Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const = 0;
};

template <int N> class TwiceDifferentiableFunction : public abstract::TwiceDifferentiableFunction {
  public:
    double evaluate(const Eigen::VectorXd &x) const final {
        if (x.size() != N) {
            throw std::invalid_argument("TwiceDifferentiableFunction::evaluate() called with "
                                        "vector of incorrect dimension");
        }
        if (!x.allFinite()) {
            throw std::invalid_argument(
                "TwiceDifferentiableFunction::evaluate() called with non-numeric vector");
        }
        return evaluateImpl(x);
    }

    int sourceDimension() const final { return N; }

    Eigen::VectorXd gradient(const Eigen::VectorXd &x) const final {
        if (x.size() != N) {
            throw std::invalid_argument("TwiceDifferentiableFunction::gradient() called with "
                                        "vector of incorrect dimension");
        }
        if (!x.allFinite()) {
            throw std::invalid_argument(
                "TwiceDifferentiableFunction::gradient() called with non-numeric vector");
        }
        Eigen::VectorXd grad = gradientImpl(x);
        if (grad.size() != N) {
            throw std::invalid_argument("gradientImpl() returned a vector of improper dimension.");
        }
        if (!grad.allFinite()) {
            throw std::invalid_argument("gradientImpl() returned a non-numeric gradient.");
        }
        return grad;
    }

    Eigen::MatrixXd hessian(const Eigen::VectorXd &x) const final {
        if (x.size() != N) {
            throw std::invalid_argument(
                "TwiceDifferentiableFunction::hessian() called with vector of incorrect dimension");
        }
        if (!x.allFinite()) {
            throw std::invalid_argument(
                "TwiceDifferentiableFunction::hessian() called with non-numeric vector");
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

  protected:
    virtual double evaluateImpl(const Eigen::VectorXd &x) const = 0;
    virtual Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const = 0;
    virtual Eigen::MatrixXd hessianImpl(const Eigen::VectorXd &x) const = 0;
};

} // namespace optim
