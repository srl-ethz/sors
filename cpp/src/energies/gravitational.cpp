#include "gravitational.h"

template<int vertexDim, int elementDim>
GravitationalEnergy<vertexDim, elementDim>::GravitationalEnergy(
    const Vector<double, vertexDim>& gravAcceleration,
    double density,
    std::array<Matrix<double, vertexDim, vertexDim>, 9>& unitMatrices)
    : ElementEnergy<vertexDim, elementDim>(unitMatrices),
      gravAcceleration_(gravAcceleration),
      density_(density) {}

template<int vertexDim, int elementDim>
double GravitationalEnergy<vertexDim, elementDim>::compute_energy(
    const Matrix<double, elementDim, vertexDim>& vertices,
    const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
    const MatrixXd& actuation) const {
    UNUSED(deformationGradients); UNUSED(actuation);

    const Vector<double, vertexDim> centerOfMass = vertices.colwise().sum() / vertices.rows();
    return -density_ * gravAcceleration_.dot(centerOfMass);
}

template<int vertexDim, int elementDim>
Vector<double, elementDim * vertexDim> GravitationalEnergy<vertexDim, elementDim>::compute_gradient(
    const Matrix<double, elementDim, vertexDim>& vertices,
    const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
    const std::vector<Matrix<double, vertexDim * vertexDim, vertexDim * elementDim>>& deformationHessians,
    const MatrixXd& actuation) const {
    UNUSED(vertices); UNUSED(deformationGradients); UNUSED(deformationHessians); UNUSED(actuation);

    Vector<double, elementDim * vertexDim> gravitationalGradient;
    for (int i = 0; i < elementDim; ++i) {
        gravitationalGradient.template segment<vertexDim>(i * vertexDim) =
            -(density_ / static_cast<double>(elementDim)) * gravAcceleration_;
    }
    return gravitationalGradient;
}

template<int vertexDim, int elementDim>
Matrix<double, elementDim * vertexDim, elementDim * vertexDim> GravitationalEnergy<vertexDim, elementDim>::compute_hessian(
    const Matrix<double, elementDim, vertexDim>& vertices,
    const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
    const std::vector<Matrix<double, vertexDim * vertexDim, vertexDim * elementDim>>& deformationHessians,
    const MatrixXd& actuation) const {
    UNUSED(vertices); UNUSED(deformationGradients); UNUSED(deformationHessians); UNUSED(actuation);

    return Matrix<double, elementDim * vertexDim, elementDim * vertexDim>::Zero();
}

template class GravitationalEnergy<3, 4>;
template class GravitationalEnergy<3, 8>;
