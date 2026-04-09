#ifndef PRESSUREFORCE_H
#define PRESSUREFORCE_H

#include "externalForce.h"

/**
 * @class PressureForce
 * @brief External surface-pressure load applied on mesh boundary faces.
 *
 * Models pressure acting on surface elements and distributes the resulting
 * normal forces to their vertices; provides the force Jacobian
 * for use in Newton-type solvers. Inherits from ExternalForce.
 */
template<int vertexDim, int elementDim>
class PressureForce : public ExternalForce<vertexDim, elementDim> {

public:
    /**
     * @brief Construct a pressure force defined on a set of surface faces.
     *
     * @param surfaceVertexIdx Matrix whose rows list the vertex indices of each
     *        surface face (e.g., 3 vertices for triangles, 4 for quads).
     * @param surfaceGroups Vector assigning each face to an actuation group,
     *        used to index into the pressure actuation vector.
     */
    PressureForce (
        MatrixXi surfaceVertexIdx,
        VectorXi surfaceGroups
    );

    // Compute pressure force, actuation is a vector of pressure values for each group
    VectorXd compute_force (const VectorXd& q, const VectorXd& actuation) const override;

    // Compute derivative of the force w.r.t. the vertex positions
    std::vector<Triplet<double>> compute_force_gradient (const VectorXd& q, const VectorXd& actuation) const override;

    // Function that returns the type of force: "pressure"
    std::string get_force_name() const override {return "pressure";}

private:
    // List of surface elements (for example (-1, 3) for triangle surface)
    const MatrixXi surfaceVertexIdx_; 
    // List of which actuation group each surface element belongs to
    const VectorXi surfaceGroups_;    
};

#endif
