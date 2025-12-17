#ifndef ELEMENTENERGY_H
#define ELEMENTENERGY_H

#include "common.h"

/**
 * @class ElementEnergy
 * @brief Abstract base class for per-element energy models in the simulation.
 *
 * Encapsulates the interface for evaluating the scalar energy, its gradient, and
 * Hessian for a single mesh element, given kinematic state, material response,
 * and optional actuation. All concrete element energy types must inherit from
 * this class and implement the virtual methods.
 */
template <int vertexDim, int elementDim>
class ElementEnergy
{
public:
    /**
     * @brief Construct an element energy model with shared precomputed data.
     *
     * Stores a set of unit matrices that
     * can be reused across derived energy implementations.
     *
     * @param unitMatrices            Collection of basis matrices used in
     *                                deformation and Hessian assembly.
     */
    ElementEnergy(std::array<Matrix<double, vertexDim, vertexDim>, 9>& unitMatrices) : unitMatrices_(unitMatrices) {};

    virtual ~ElementEnergy() = default; // Make destructor virtual

    /**
     * @brief Compute the scalar energy contribution of a single element.
     *
     * Evaluates the element’s potential (and, where applicable, inertial or
     * actuation-related) energy given the current kinematic state, material
     * parameters, and time-stepping scheme.
     *
     * @param vertices            Current element vertex positions.
     * @param deformationGradients Precomputed deformation gradients for the element.
     * @param actuation           Actuation state relevant to this energy (may be empty).
     * @return Scalar energy value contributed by this element.
     */
    virtual double compute_energy(
        const Matrix<double, elementDim, vertexDim>& vertices, 
        const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
        const MatrixXd& actuation) const = 0;

    /**
     * @brief Compute the gradient of the element energy with respect to vertex positions.
     *
     * Returns the first derivative of the scalar element energy w.r.t. the stacked
     * vertex coordinates, suitable for assembling the global residual vector in a
     * Newton-type or gradient-based solver.
     *
     * @param vertices            Current element vertex positions.
     * @param deformationGradients Precomputed deformation gradients for the element.
     * @param deformationHessians  Precomputed derivatives of deformation gradients
     * @param actuation           Actuation state relevant to this energy (may be empty).
     * @return Stacked gradient vector dE/dx for this element.
     */
    virtual Vector<double, elementDim*vertexDim> compute_gradient(
        const Matrix<double, elementDim, vertexDim>& vertices, 
        const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
        const std::vector<Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>>& deformationHessians,
        const MatrixXd& actuation) const = 0;

    /**
     * @brief Compute the Hessian of the element energy with respect to vertex positions.
     *
     * Returns the second derivative of the scalar element energy w.r.t. the stacked
     * vertex coordinates, used for assembling the global tangent matrix in implicit
     * solvers.
     *
     * @param vertices            Current element vertex positions.
     * @param deformationGradients Precomputed deformation gradients for the element.
     * @param deformationHessians  Precomputed derivatives of deformation gradients
     * @param actuation           Actuation state relevant to this energy (may be empty).
     * @return Hessian matrix d²E/dx² for this element.
     */
    virtual Matrix<double, elementDim*vertexDim, elementDim*vertexDim> compute_hessian(
        const Matrix<double, elementDim, vertexDim>& vertices, 
        const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
        const std::vector<Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>>& deformationHessians,
        const MatrixXd& actuation) const = 0;

    /**
     * @brief Update internal, state-dependent quantities of the energy model.
     *
     * Default implementation is a no-op. Derived classes can override this to
     * advance internal variables (e.g., history-dependent muscle or viscoelastic
     * states) using the current configuration, acceleration, time step, and
     * deformation gradients.
     *
     * @param vertices             Current element vertex positions.
     * @param dt                   Time step size.
     * @param deformationGradients Precomputed deformation gradients for the element.
     */
    virtual void update_state(
        const Matrix<double, elementDim, vertexDim>& vertices, 
        const double dt,
        const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients){
            UNUSED(vertices); UNUSED(dt); UNUSED(deformationGradients); return;};
   
    // Returns the name of the energy
    virtual std::string get_energy_name() const = 0;

    // Flag to indicate if the actuation is active (default is false)
    bool actuationFlag_ = false; 

protected:   
    // Unit matrices of vertexDim x vertexDim            
    const std::array<Matrix<double, vertexDim, vertexDim>, 9> unitMatrices_;   
};

#endif
