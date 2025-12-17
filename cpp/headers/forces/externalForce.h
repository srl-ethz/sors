#ifndef EXTERNALFORCES_H
#define EXTERNALFORCES_H

#include "common.h"

/**
 * @class ExternalForce
 * @brief Abstract base class for external force models applied to the system.
 *
 * Defines the interface for all external force types (e.g., pressure, vertex forces)
 * that contribute to the global force/gradient in the variational formulation.
 */
template<int vertexDim, int elementDim>
class ExternalForce {
public:
    ExternalForce () {}

    /**
     * @brief Compute the external force vector for the current configuration.
     *
     * @param vertices Flattened vertex positions of the system (size 3n).
     * @param actuation Actuation/state vector associated with this force (e.g., pressures, amplitudes).
     * @return Vector of external forces to be added to the global gradient (size 3n).
     */
    virtual VectorXd compute_force (const VectorXd& vertices, const VectorXd& actuation) const = 0;

    /**
     * @brief Compute the Jacobian of the external force w.r.t. vertex positions (optional).
     *
     * Default implementation returns an empty triplet list; derived classes may override
     * this to provide df/dx for use in the Newton solve.
     *
     * @param q Flattened vertex positions of the system (size 3n).
     * @param actuation Actuation/state vector associated with this force.
     * @return Triplet list representing the sparse force Jacobian matrix.
     */
    virtual std::vector<Triplet<double>> compute_force_gradient (const VectorXd& q, const VectorXd& actuation) const {
        UNUSED(q); UNUSED(actuation);
        std::vector<Triplet<double>> forceDerivativeTriplets; // Entries into the sparse hessian of shape [q.size(), q.size()]
        return forceDerivativeTriplets; // Not implemented yet, return empty vector
    }

    // Function that returns the type of force: "pressure", "vertexForce"
    virtual std::string get_force_name() const = 0;

    // Flag to indicate if the actuation is active (default is false)
    bool actuationFlag_ = false; 
};

#endif