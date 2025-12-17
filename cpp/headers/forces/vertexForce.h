#ifndef VERTEXFORCE_H
#define VERTEXFORCE_H

#include "externalForce.h"

/**
 * @class VertexForce
 * @brief Per-vertex external force field added directly to the system.
 *
 * Wraps a user-provided force vector (size 3n) that specifies arbitrary
 * forces on each vertex DOF and injects it into the global residual.
 * Inherits from ExternalForce.
 */
template<int vertexDim, int elementDim>
class VertexForce : public ExternalForce<vertexDim, elementDim> {

public:
    /**
     * @brief Construct a vertex-based external force source.
     *
     * Marks this force as actuated, so the solver expects an actuation vector
     * containing the per-vertex force entries (size 3 * numVertices).
     */
    VertexForce () {this->actuationFlag_ = true;};

    // Compute the external force vector for the current configuration
    VectorXd compute_force (const VectorXd& q, const VectorXd& actuation) const override;

    // Function that returns the type of force
    std::string get_force_name() const override {return "vertexForce";}
};

template class VertexForce<3, 4>;
template class VertexForce<3, 8>;

#endif
