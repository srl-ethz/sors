#ifndef STABLENEOHOOKEAN_H
#define STABLENEOHOOKEAN_H

#include "elementEnergy.h"

/**
 * @class StableNeoHookeanEnergy
 * @brief Stable Neo-Hookean elastic energy for robust flesh/soft-tissue simulation.
 *
 * Implements the stable Neo-Hookean model of Smith et al. 
 * (“Stable neo-hookean flesh simulation”, TOG 2018), providing
 * energy, gradient, and Hessian for a single finite element and 
 * inheriting from ElementEnergy.
 */
template<int vertexDim, int elementDim>
class StableNeoHookeanEnergy : public ElementEnergy<vertexDim, elementDim> {

public: 
    /**
     * @brief Construct a stable Neo-Hookean energy term from material parameters.
     *
     * Reparameterizes Young’s modulus and Poisson’s ratio into the stabilized Lamé
     * parameters (mu, lambda, alpha).
     */
    StableNeoHookeanEnergy(
        double youngsModulus,
        double poissonsRatio,
        std::array<Matrix<double, vertexDim, vertexDim>, 9>& unitMatrices);

    // Compute stable neohookean elastic energy
    double compute_energy(
        const Matrix<double, elementDim, vertexDim>& vertices, 
        const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
        const MatrixXd& actuation) const override;

    // Compute gradient of stable neohookean elastic energy
    Vector<double, elementDim*vertexDim> compute_gradient(
        const Matrix<double, elementDim, vertexDim>& vertices, 
        const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
        const std::vector<Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>>& deformationHessians,
        const MatrixXd& actuation) const override;

    // Compute hessian of stable neohookean elastic energy 
    Matrix<double, elementDim*vertexDim, elementDim*vertexDim> compute_hessian(
        const Matrix<double, elementDim, vertexDim>& vertices, 
        const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
        const std::vector<Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>>& deformationHessians,
        const MatrixXd& actuation) const override;

    // Returns the name of the energy
    virtual std::string get_energy_name() const override {
        return "stableneohookean";
    };

private:
    // Material parameters
    const double youngsModulus_;
    const double poissonsRatio_;
    const double mu_; 
    const double lambda_; 
    const double alpha_;
};

#endif
