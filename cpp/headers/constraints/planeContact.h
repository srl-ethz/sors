#ifndef PLANECONTACT_H
#define PLANECONTACT_H

#include "constraint.h"
#include "plane3D.h"


/**
 * @class PlaneContact
 * @brief Inequality contact constraint enforcing non-penetration with planar obstacles.
 *
 * Inherits from Constraint. For a selected set of surface vertices, the constraint
 * uses the signed distance to one of several planes to enforce g_k(x) = n·(x − r0) ≥ 0,
 * where n is the plane normal and r0 a point on the plane.
 */
template<int vertexDim, int elementDim>
class PlaneContact : public Constraint<vertexDim, elementDim> {

public:
    /**
     * @brief Construct a plane contact constraint.
     *
     * Initializes the list of contact planes, the indices of surface vertices to monitor,
     * and a collision mask that maps each surface vertex either to a colliding plane index
     * or to -1 if it is currently inactive.
     *
     * @param numVertices        Total number of vertices in the system.
     * @param planes             Collection of Plane3D objects defining contact planes.
     * @param surfaceVerticesIdx Indices of vertices on the object's surface to be tested for contact.
     */
    PlaneContact(int numVertices, std::vector<Plane3D> planes, VectorXi surfaceVerticesIdx)
    : Constraint<vertexDim, elementDim>(numVertices), 
      planes_(planes), 
      surfaceVerticesIdx_(surfaceVerticesIdx),
      planeCollisionMask_(std::vector<int>(surfaceVerticesIdx_.size(), -1)) {}
    
    // Computes the scalar constraint function 
    VectorXd compute_constraint(const VectorXd& vertices, const VectorXd& verticesPrev, double dt) const override;

    // Computes the gradient of the scalar constraint function 
    std::vector<Triplet<double>> compute_constraint_gradient(const VectorXd& vertices, const VectorXd& verticesPrev, double dt) const override;

    // Updates the active constraints
    bool update_active_constraints(const VectorXd& vertices) override;

    // Returns the type of constraint: "equality" or "inequality"
    std::string get_constraint_type() const override {
        return "inequality";
    }

    // Returns the name of the constraint
    std::string get_constraint_name() const override {
        return "planeContact";
    }

    // Sets new planes
    void set_planes(const std::vector<Plane3D>& planes) {
        planes_ = planes;
        // Reset the planeCollisionMask_ to -1
        planeCollisionMask_.assign(surfaceVerticesIdx_.size(), -1);
    }

protected:
    // Collection of planes
    std::vector<Plane3D> planes_;
    // Vector of surface vertices indices of object
    VectorXi surfaceVerticesIdx_;
    // Vector size of surfaceVerticesIdx_ that stores -1 if no collision, plane index otherwise for each surface vertex
    std::vector<int> planeCollisionMask_;
};

#endif