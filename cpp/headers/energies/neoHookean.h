#ifndef NEOHOOKEAN_H
#define NEOHOOKEAN_H

#include "elementEnergy.h"

/**
 * @class NeoHookeanEnergy
 * @brief Compressible Neo-Hookean elastic energy for a single finite element.
 *
 * Inherits from ElementEnergy and models isotropic hyperelastic behavior
 * parameterized by Young's modulus and Poisson's ratio.
 */
template<int vertexDim, int elementDim>
class NeoHookeanEnergy : public ElementEnergy<vertexDim, elementDim> {

public: 
    /**
     * @brief Construct a Neo-Hookean energy term from material and global parameters.
     *
     * Initializes Lamé parameters from Young's modulus and Poisson's ratio and
     * forwards unit matrices to the ElementEnergy base class.
     */
    NeoHookeanEnergy(
        double youngsModulus,
        double poissonsRatio,
        std::array<Matrix<double, vertexDim, vertexDim>, 9>& unitMatrices);

    // Compute neohookean elastic energy
    double compute_energy(
        const Matrix<double, elementDim, vertexDim>& vertices, 
        const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
        const MatrixXd& actuation) const override;

    // Compute gradient of neohookean elastic energy
    Vector<double, elementDim*vertexDim> compute_gradient(
        const Matrix<double, elementDim, vertexDim>& vertices, 
        const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
        const std::vector<Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>>& deformationHessians,
        const MatrixXd& actuation) const override;

    // Compute hessian of neohookean elastic energy 
    Matrix<double, elementDim*vertexDim, elementDim*vertexDim> compute_hessian(
        const Matrix<double, elementDim, vertexDim>& vertices, 
        const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
        const std::vector<Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>>& deformationHessians,
        const MatrixXd& actuation) const override;

    // Returns the name of the energy
    std::string get_energy_name() const override {
        return "neohookean";
    }

private:
    // Material parameters
    const double youngsModulus_;
    const double poissonsRatio_;
    const double lambda_; 
    const double mu_; 
};

#endif
