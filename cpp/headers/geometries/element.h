#ifndef ELEMENT_H
#define ELEMENT_H

#include "common.h"
#include "params.h"
#include "elementEnergy.h"
#include "neoHookean.h"
#include "stableNeoHookean.h"
#include "pseudostrain.h"
#include "softconMuscle.h"
#include "gravitational.h"

/**
 * @class Element
 * @brief Generic linear finite element wrapper combining multiple energy terms.
 *
 * Stores undeformed geometry, density, and quadrature data for a single element
 * and provides common routines to evaluate total energy, gradient, and Hessian
 * by aggregating all attached ElementEnergy instances.
 */
template<int vertexDim, int elementDim>
class Element
{

public:
    /**
     * @brief Construct an element from undeformed vertices, density, and volume.
     *
     * Initializes constant element properties such as mass, undeformed geometry,
     * density, volume, and the number of quadrature points used in integration.
     */
    Element(const Matrix<double, elementDim, vertexDim>& undeformedVertices, double density, double undeformedVolume, int nquadpoints) : mass_(density * undeformedVolume), undeformedVertices_(undeformedVertices), undeformedDensity_(density), undeformedVolume_(undeformedVolume), nquadpoints_(nquadpoints) {};
  
    /**
     * @brief Compute the total element energy by summing all attached energy terms.
     *
     * Evaluates each ElementEnergy for the current kinematics and time-stepping
     * scheme, multiplies by the undeformed volume, and returns the accumulated
     * energy for this element.
     */
    virtual double compute_energy(
        Matrix<double, elementDim, vertexDim>& vertices,
        Params& actuation) const {
        
        // Compute element integral data (deformationGradient F is the only one that changes over time)
        auto deformationGradients = compute_deformation_gradients(vertices);

            double totalEnergy = 0.0;
            // Compute energies
            for (std::size_t i = 0; i < elementEnergies_.size(); i++) {
                // Find actuation where needed
                MatrixXd act; 
                if (this->elementEnergies_[i]->actuationFlag_) act = actuation.get_value(this->elementEnergies_[i]->get_energy_name()); 
                else act = MatrixXd::Zero(0, 0); 
                totalEnergy += elementEnergies_[i]->compute_energy(vertices, deformationGradients, act) * undeformedVolume_;
            }
        
        return totalEnergy;
    }

    /**
     * @brief Compute the gradient of the total element energy w.r.t. vertex DOFs.
     *
     * Evaluates and sums the gradients from all attached ElementEnergy instances
     * for the current kinematics and time-stepping scheme.
     */
    virtual Vector<double, vertexDim*elementDim> compute_gradient(
        Matrix<double, elementDim, vertexDim>& vertices,
        Params& actuation) const = 0;

    /**
     * @brief Compute the Hessian of the total element energy w.r.t. vertex DOFs.
     *
     * Evaluates and sums the Hessians from all attached ElementEnergy instances
     * for the current kinematics and time-stepping scheme.
     */
    virtual Matrix<double, vertexDim*elementDim, vertexDim*elementDim> compute_hessian(
        Matrix<double, elementDim, vertexDim>& vertices,
        Params& actuation) const = 0;

    /**
     * @brief Compute the contribution of a single energy type for visualization.
     *
     * Evaluates only the specified energy term (if present on this element) for
     * the current configuration and returns its total contribution.
     */
    virtual double compute_specific_energy(
        Matrix<double, elementDim, vertexDim>& vertices,
        Params& actuation, std::string energyType="") const {

        // Compute element's specific properties
        auto deformationGradients = compute_deformation_gradients(vertices);

        double totalEnergy = 0.0;
        auto search = std::find(elementEnergiesString_.begin(), elementEnergiesString_.end(), energyType);
        if(search != std::end(elementEnergiesString_)){
            // If the required energy type is assigned to the corresponding tetrahedron element, 
            // find actuation where needed.
            int i = search-elementEnergiesString_.begin();
            MatrixXd act; 
            if (this->elementEnergies_[i]->actuationFlag_) act = actuation.get_value(this->elementEnergies_[i]->get_energy_name()); 
            else act = MatrixXd::Zero(0, 0); 
            totalEnergy = elementEnergies_[i]->compute_energy(vertices, deformationGradients, act) * undeformedVolume_;
        }
        return totalEnergy;
    }

    /**
     * @brief Update internal state of all attached energy models for this element.
     *
     * Recomputes deformation gradients and forwards them, together with the
     * current acceleration and time step, to each ElementEnergy::update_state().
     */
    void update_state(
        Matrix<double, elementDim, vertexDim>& vertices,
        double dt) {

        auto deformationGradients = compute_deformation_gradients(vertices);
        for (unsigned int i = 0; i < elementEnergies_.size(); i++)
        {
            // Update energy related state parameters
            elementEnergies_[i]->update_state(vertices, dt, deformationGradients);
        }
    }

    /**
     * @brief Compute the shortest edge length of the element in the current state.
     *
     * Iterates over all vertex pairs of the element and returns the minimum
     * Euclidean distance, useful for CFL or quality checks.
     */
    virtual double compute_minimal_edge_length(Matrix<double, elementDim, vertexDim>& vertices) const {
        double minimalEdgeLength = -1.0;
        for (int i = 0; i < elementDim; i++) {
            for (int j = i + 1; j < elementDim; j++) {
                double edgeLength = (vertices.row(i) - vertices.row(j)).norm();
                if (minimalEdgeLength < 0.0 || edgeLength < minimalEdgeLength) {
                    minimalEdgeLength = edgeLength;
                }
            }
        }
        return minimalEdgeLength;
    }

    // Returns mass of the element
    double compute_mass () const {
        return mass_;
    }

    // Returns volume of the element
    double get_volume (const Matrix<double, elementDim, vertexDim>& vertices) const {
        return compute_volume(vertices);
    }

    // Returns undeformed volume of the element
    double get_undeformed_volume () const {
        return undeformedVolume_;
    }

    // Returns undeformed vertices of the element
    Matrix<double, elementDim, vertexDim> get_undeformed_vertices() const {
        return undeformedVertices_;
    }

    // Returns deformation hessian of the element
    Matrix<double, vertexDim * vertexDim, elementDim * vertexDim> get_deformation_hessian() const {
        // Return average deformation hessian among quadrature points
        Matrix<double, vertexDim * vertexDim, elementDim * vertexDim> avgDeformationHessian = Matrix<double, vertexDim * vertexDim, elementDim * vertexDim>::Zero();
        for (const auto& defHess : deformationHessians_) {
            avgDeformationHessian += defHess;
        }
        avgDeformationHessian /= deformationHessians_.size();
        return avgDeformationHessian;
    }
    // Mass of the element 
    const double mass_; 

protected:

    const Matrix<double, elementDim, vertexDim> undeformedVertices_;
    const double undeformedDensity_; 
    const double undeformedVolume_; 
    // Number of quadrature points used for the energy integral
    const int nquadpoints_;
    
    // The deformation hessians: The gradient w.r.t the x_j (deformed nodes) of F
    std::vector<Eigen::Matrix<double, vertexDim * vertexDim, vertexDim * elementDim>> deformationHessians_;
    // List of energy types for this element 
    std::vector<std::string> elementEnergiesString_;
    // List energy objects for different energy types;          
    std::vector<std::unique_ptr<ElementEnergy<vertexDim, elementDim>>> elementEnergies_;

    // Computes density of the element
    double compute_density(const Matrix<double, elementDim, vertexDim>& vertices) const {
        return compute_volume(vertices) / mass_;
    }

    // Computes volume of the element
    virtual double compute_volume(const Matrix<double, elementDim, vertexDim>& vertices) const = 0;

    // Computes deformation gradients of the element over the quadrature points
    virtual std::vector<Matrix<double, vertexDim, vertexDim>> compute_deformation_gradients(Matrix<double, elementDim, vertexDim>& vertices) const = 0;

    // Computes center of mass of element
    virtual Vector<double, vertexDim> compute_com(const Matrix<double, elementDim, vertexDim>& vertices) const = 0;
};

#endif
