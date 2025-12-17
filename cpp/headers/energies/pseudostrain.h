#ifndef PSEUDOSTRAIN_H
#define PSEUDOSTRAIN_H

#include "elementEnergy.h"

/**
 * @class PseudostrainEnergy
 * @brief Pseudostrain-based hyperelastic energy for non-contracting myocardium.
 *
 * Inherits from ElementEnergy and implements the deviatoric–volumetric
 * pseudostrain formulation proposed in Guccione et al. (1991)
 * https://doi.org/10.1115/1.2891193 for modeling passive myocardial tissue.
 */
template<int vertexDim, int elementDim>
class PseudostrainEnergy : public ElementEnergy<vertexDim, elementDim> {

public: 
    /**
     * @brief Construct a pseudostrain energy term from material and global parameters.
     *
     * Initializes the pseudostrain coefficients b, c, D and forwards unit matrices to the ElementEnergy base class.
     */
    PseudostrainEnergy(
        double b, double c, double D,
        std::array<Matrix<double, vertexDim, vertexDim>, 9>& unitMatrices);

    // Compute pseudostrain elastic energy
    double compute_energy(
        const Matrix<double, elementDim, vertexDim>& vertices, 
        const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
        const MatrixXd& actuation) const override;

    // Compute gradient of pseudostrain elastic energy
    Vector<double, elementDim*vertexDim> compute_gradient(
        const Matrix<double, elementDim, vertexDim>& vertices, 
        const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
        const std::vector<Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>>& deformationHessians,
        const MatrixXd& actuation) const override;

    // Compute hessian of pseudostrain elastic energy 
    Matrix<double, elementDim*vertexDim, elementDim*vertexDim> compute_hessian(
        const Matrix<double, elementDim, vertexDim>& vertices, 
        const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
        const std::vector<Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>>& deformationHessians,
        const MatrixXd& actuation) const override;

    // Function that returns the name of the energy
    std::string get_energy_name() const override {
        return "pseudostrain";
    }

private:
    // Material parameters
    const double b_;
    const double c_;
    const double D_; 

    // Helper functions for energy, gradient, and hessian computations
    double compute_deviatoric_energy(const Matrix<double, vertexDim, vertexDim>& deformationGradient) const;
    double compute_volume_energy(const Matrix<double, vertexDim, vertexDim>& deformationGradient) const;    
    Vector<double, elementDim*vertexDim> compute_deviatoric_gradient(const Matrix<double, vertexDim, vertexDim>& deformationGradient, const Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>& deformationHessian) const;
    Vector<double, elementDim*vertexDim> compute_volume_gradient(const Matrix<double, vertexDim, vertexDim>& deformationGradient, const Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>& deformationHessian) const;
    Matrix<double, elementDim*vertexDim, elementDim*vertexDim> compute_deviatoric_hessian(const Matrix<double, vertexDim, vertexDim>& deformationGradient, const Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>& deformationHessian) const;
    Matrix<double, elementDim*vertexDim, elementDim*vertexDim> compute_volume_hessian(const Matrix<double, vertexDim, vertexDim>& deformationGradient, const Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>& deformationHessian) const;      
};

#endif