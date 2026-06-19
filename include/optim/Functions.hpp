#pragma once
#include <Eigen/Dense>
#include <stdexcept>

namespace optim {

class Function {
  public:
    virtual ~Function() = default;

    double evaluate(const Eigen::VectorXd &x) const {
        if (x.size() != sourceDimension())
            throw std::invalid_argument("evaluate() called with vector of incorrect dimension");
        if (!x.allFinite())
            throw std::invalid_argument("evaluate() called with non-numeric vector");
        return evaluateImpl(x);
    }

    int sourceDimension() const { return n_; }

  protected:
    explicit Function(int n) : n_(n) {}
    virtual double evaluateImpl(const Eigen::VectorXd &x) const = 0;

  private:
    int n_;
};

class DifferentiableFunction : public Function {
  public:
    Eigen::VectorXd gradient(const Eigen::VectorXd &x) const {
        if (x.size() != sourceDimension())
            throw std::invalid_argument("gradient() called with vector of incorrect dimension");
        if (!x.allFinite())
            throw std::invalid_argument("gradient() called with non-numeric vector");
        Eigen::VectorXd grad = gradientImpl(x);
        if (grad.size() != sourceDimension())
            throw std::invalid_argument("gradientImpl() returned a vector of improper dimension.");
        if (!grad.allFinite())
            throw std::invalid_argument("gradientImpl() returned a non-numeric gradient.");
        return grad;
    }

  protected:
    explicit DifferentiableFunction(int n) : Function(n) {}
    virtual Eigen::VectorXd gradientImpl(const Eigen::VectorXd &x) const = 0;
};

class TwiceDifferentiableFunction : public DifferentiableFunction {
  public:
    Eigen::MatrixXd hessian(const Eigen::VectorXd &x) const {
        if (x.size() != sourceDimension())
            throw std::invalid_argument("hessian() called with vector of incorrect dimension");
        if (!x.allFinite())
            throw std::invalid_argument("hessian() called with non-numeric vector");
        Eigen::MatrixXd hess = hessianImpl(x);
        if (hess.rows() != sourceDimension() || hess.cols() != sourceDimension())
            throw std::invalid_argument("hessianImpl() returned a matrix of improper dimension.");
        if (!hess.allFinite())
            throw std::invalid_argument("hessianImpl() returned a non-numeric matrix.");
        return hess;
    }

  protected:
    explicit TwiceDifferentiableFunction(int n) : DifferentiableFunction(n) {}
    virtual Eigen::MatrixXd hessianImpl(const Eigen::VectorXd &x) const = 0;
};

} // namespace optim
