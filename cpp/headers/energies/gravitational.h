#ifndef GRAVITATIONAL_H
#define GRAVITATIONAL_H

#include "elementEnergy.h"

template<int vertexDim, int elementDim>
class GravitationalEnergy : public ElementEnergy<vertexDim, elementDim> {
public:
    GravitationalEnergy(
        const Vector<double, vertexDim>& gravAcceleration,
        double density,
        std::array<Matrix<double, vertexDim, vertexDim>, 9>& unitMatrices);

    double compute_energy(
        const Matrix<double, elementDim, vertexDim>& vertices,
        const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
        const MatrixXd& actuation) const override;

    Vector<double, elementDim * vertexDim> compute_gradient(
        const Matrix<double, elementDim, vertexDim>& vertices,
        const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
        const std::vector<Matrix<double, vertexDim * vertexDim, vertexDim * elementDim>>& deformationHessians,
        const MatrixXd& actuation) const override;

    Matrix<double, elementDim * vertexDim, elementDim * vertexDim> compute_hessian(
        const Matrix<double, elementDim, vertexDim>& vertices,
        const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
        const std::vector<Matrix<double, vertexDim * vertexDim, vertexDim * elementDim>>& deformationHessians,
        const MatrixXd& actuation) const override;

    std::string get_energy_name() const override {
        return "gravitational";
    }

private:
    const Vector<double, vertexDim> gravAcceleration_;
    const double density_;
};

#endif
