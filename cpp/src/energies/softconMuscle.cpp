#include "softconMuscle.h"

template <int vertexDim, int elementDim>
SoftconMuscleEnergy<vertexDim, elementDim>::SoftconMuscleEnergy(
    int muscleGroup,
    double muscleStiffness, VectorXd muscleDirection,
    std::array<Matrix<double, vertexDim, vertexDim>, 9>& unitMatrices) : ElementEnergy<vertexDim, elementDim>(unitMatrices), muscleGroup_(muscleGroup), k_(muscleStiffness), m_(muscleDirection), F_(Matrix<double, vertexDim, vertexDim>::Zero()) {
        this->actuationFlag_ = true; // Softcon muscle energy has actuation flag set to true
    }

template <int vertexDim, int elementDim>
double SoftconMuscleEnergy<vertexDim, elementDim>::compute_energy(
    const Matrix<double, elementDim, vertexDim>& vertices, 
    const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
    const MatrixXd& actuation) const {

    UNUSED(vertices); 

    // Normalized quadrature weights:
    int nquadpoints = deformationGradients.size();
    double quadratureWeights = 1. / nquadpoints; 

    // Initialize energy
    double elasticStrainEnergy = 0;

    // Compute integral over the element
    for (int s = 0; s < nquadpoints; ++s) {
        const Matrix<double, vertexDim, vertexDim>& F = deformationGradients[s];
        double e = actuation.coeff(this->muscleGroup_);
        double r = (1 - e) / (F * m_).norm();
        elasticStrainEnergy += (k_ / 2 * ((1 - r) * F * m_).norm()) * quadratureWeights;
    }
    return elasticStrainEnergy; 
}


template <int vertexDim, int elementDim>
Vector<double, elementDim * vertexDim> SoftconMuscleEnergy<vertexDim, elementDim>::compute_gradient(
    const Matrix<double, elementDim, vertexDim>& vertices, 
    const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
    const std::vector<Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>>& deformationHessians,
    const MatrixXd& actuation) const {

    UNUSED(vertices); 

    // Normalized quadrature weights:
    int nquadpoints = deformationGradients.size();
    double quadratureWeights = 1. / nquadpoints; 
    
    Vector<double, vertexDim * elementDim> einsteinTerms_total = Vector<double, vertexDim * elementDim>::Zero();

    // Compute the integral over the element
    for (int s = 0; s < nquadpoints; ++s) {
        const Matrix<double, vertexDim, vertexDim>& F = deformationGradients[s];
        const Matrix<double, vertexDim * vertexDim, vertexDim * elementDim>& nablaF = deformationHessians[s];
        double e = actuation.coeff(this->muscleGroup_);
        double l = (F * m_).norm();
        double r = (1 - e) / l;

        Vector<double, vertexDim * elementDim> einsteinTerms = Vector<double, vertexDim * elementDim>::Zero();
        // m_c * F_ab * m_b * nablaF_aci
        for (int a = 0; a < vertexDim; ++a) {
            for (int b = 0; b < vertexDim; ++b) {
                for (int c = 0; c < vertexDim; ++c) {
                    for (int i = 0; i < vertexDim * elementDim; ++i) {
                        int ac = a * vertexDim + c;
                        einsteinTerms(i) += m_(c) * F(a, b) * m_(b) * nablaF(ac, i);
                    }
                }
            }
        }
        einsteinTerms *= k_ * (1 - r);
        einsteinTerms_total += einsteinTerms * quadratureWeights;
    }
    return einsteinTerms_total;
}


template <int vertexDim, int elementDim>
Matrix<double, elementDim * vertexDim, elementDim * vertexDim> SoftconMuscleEnergy<vertexDim, elementDim>::compute_hessian(
    const Matrix<double, elementDim, vertexDim>& vertices, 
    const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
    const std::vector<Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>>& deformationHessians,
    const MatrixXd& actuation) const {

    UNUSED(vertices); 

    Matrix<double, elementDim * vertexDim, elementDim * vertexDim> energyHessian = Matrix<double, elementDim * vertexDim, elementDim * vertexDim>::Zero();

    // Normalized quadrature weights:
    int nquadpoints = deformationGradients.size();
    double quadratureWeights = 1. / nquadpoints; 
    
    // Computing the integral over the quadrature points
    for (int s = 0; s < nquadpoints; ++s) {
        const Matrix<double, vertexDim, vertexDim>& F = deformationGradients[s];
        const Matrix<double, vertexDim * vertexDim, vertexDim * elementDim>& nablaF = deformationHessians[s];
        double e = actuation.coeff(this->muscleGroup_);
        double l = (F * m_).norm();
        double r = (1 - e) / l;

        Vector<double, vertexDim * elementDim> preBigTermOne = Vector<double, vertexDim * elementDim>::Zero();
        // m_c * F_ab * m_b * nablaF_aci
        for (int a = 0; a < vertexDim; ++a) {
            for (int b = 0; b < vertexDim; ++b) {
                for (int c = 0; c < vertexDim; ++c) {
                    for (int i = 0; i < vertexDim * elementDim; ++i) {
                        int ac = a * vertexDim + c;
                        preBigTermOne(i) += m_(c) * F(a, b) * m_(b) * nablaF(ac, i);
                    }
                }
            }
        }

        Vector<double, vertexDim * elementDim> preBigTermTwo = Vector<double, vertexDim * elementDim>::Zero();
        // nablaF_aci * m_c
        for (int a = 0; a < vertexDim; ++a) {
            for (int c = 0; c < vertexDim; ++c) {
                for (int i = 0; i < vertexDim * elementDim; ++i) {
                    int ac = a * vertexDim + c;
                    preBigTermTwo(i) += nablaF(ac, i) * m_(c);
                }
            }
        }
        // Getting the outer products
        Matrix<double, vertexDim * elementDim, vertexDim * elementDim> bigTermOne = (r / (l * l)) * preBigTermOne * preBigTermOne.transpose();
        Matrix<double, vertexDim * elementDim, vertexDim * elementDim> bigTermTwo = (1 - r) * preBigTermTwo * preBigTermTwo.transpose();

        energyHessian +=  k_ * (bigTermOne + bigTermTwo) * quadratureWeights;
    }
    return energyHessian;
}

template class SoftconMuscleEnergy<3, 4>;
template class SoftconMuscleEnergy<3, 8>;
