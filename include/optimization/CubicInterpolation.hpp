#include "LineSearch.hpp"

class CubicInterpolation : public LineSearch {
public:
    CubicInterpolation(); 

    double chooseStep(
        ObjectiveFunction& f, 
        const Eigen::VectorXd& x,
        const Eigen::VectorXd& direction,
        const Eigen::VectorXd& gradient
    ) override;
};