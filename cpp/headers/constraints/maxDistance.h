#ifndef MAXDISTANCE_H
#define MAXDISTANCE_H

#include "constraint.h"

/**
 * @class MaxDistance
 * @brief Inequality constraint enforcing an upper bound on the distance between vertex pairs.
 *
 * Inherits from Constraint. Each constraint enforces ||x_i − x_j|| ≤ d_max for a specified
 * pair of vertices. All max-distance constraints are fixed for the entire simulation and 
 * never update their active set.
 */
template<int vertexDim, int elementDim>
class MaxDistance : public Constraint<vertexDim, elementDim> {

public:
    /**
     * @brief Construct a max-distance constraint set.
     *
     * Initializes a list of vertex pairs (i, j) with an associated maximum allowed distance.
     * All constraints are marked as active and fixed.
     *
     * @param numVertices  Total number of vertices in the system.
     * @param maxDistanceConstraintPairs  List of (vertex1, vertex2, maxDistance) tuples.
     */
    MaxDistance(int numVertices, std::vector<std::tuple<int, int, double>> maxDistanceConstraintPairs)
    : Constraint<vertexDim, elementDim>(numVertices), 
      maxDistanceConstraintPairs_(maxDistanceConstraintPairs) {
        // Set number of active constraints
        this->set_num_active_constraints(static_cast<int>(maxDistanceConstraintPairs_.size()));
        // Set the fixed constraint flag to true, since max distance constraints are always fixed
        this->set_fixed_constraint_flag(true);
    }
    
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
        return "maxDistance";
    }

protected:
    // Vector (size of current active vertices) that stores a list of tuples (vertex1, vertex2, maxDistance) 
    std::vector<std::tuple<int, int, double>> maxDistanceConstraintPairs_; 
};

#endif