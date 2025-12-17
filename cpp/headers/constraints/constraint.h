#ifndef CONSTRAINT_H
#define CONSTRAINT_H

#include "common.h"

/**
 * @class Constraint
 * @brief Abstract base class for all constraint types in the simulation framework.
 *
 * Constraints define scalar functions g(x) that restrict the motion of vertices.
 * Inequality constraints satisfy g(x) ≥ 0; equality constraints satisfy g(x) = 0.
 * Each derived constraint class must implement the virtual methods for evaluating
 * constraint values, gradients, and active-set updates.
 */
template<int vertexDim, int elementDim>
class Constraint {

public:
    // Construct a constraint object tied to a system of numVertices vertices.
    Constraint(int numVertices)
    : numVertices_(numVertices) {}
  
    /**
     * @brief Evaluate all active scalar constraint functions.
     *
     * Must be implemented by derived classes. Returns a vector g(x) containing one
     * value per active constraint. For inequality constraints, g(x) ≥ 0 defines the
     * feasible region; for equality constraints, g(x) = 0 must hold exactly.
     *
     * @param vertices      Current system state (3n vector).
     * @param verticesPrev  Previous system state (3n vector), may be unused.
     * @param dt            Timestep, may be unused.
     * @return VectorXd     Constraint values for all active constraints.
     */ 
    virtual VectorXd compute_constraint(const VectorXd& vertices, const VectorXd& verticesPrev, double dt) const = 0;

    /**
     * @brief Compute gradients of all active constraint functions.
     *
     * Must be implemented by derived classes. Produces a sparse Jacobian ∂g/∂x in
     * triplet format with one row per active constraint. Only the derivatives with
     * respect to vertices involved in the constraint are populated.
     *
     * @param vertices      Current system state (3n vector).
     * @param verticesPrev  Previous system state (3n vector), may be unused.
     * @param dt            Timestep, may be unused.
     * @return std::vector<Triplet<double>>  Sparse Jacobian entries.
     */
    virtual std::vector<Triplet<double>> compute_constraint_gradient(const VectorXd& vertices, const VectorXd& verticesPrev, double dt) const = 0;

    /**
     * @brief Update the active-set structure for this constraint type.
     *
     * Must be implemented by derived classes. Determines which constraints are
     * currently active based on the simulated state. Updates the internal count
     * of active constraints and returns whether the update succeeded.
     *
     * @param vertices  Current system state (3n vector).
     * @return bool     True if updated successfully; false on error.
     */
    virtual bool update_active_constraints(const VectorXd& vertices) = 0;


    // Returns the type of constraint: "equality" or "inequality"
    virtual std::string get_constraint_type() const = 0;
    // Returns the name of the constraint
    virtual std::string get_constraint_name() const = 0;
    // Get number of vertices
    int get_num_vertices() const {return numVertices_;}
    // Get number of active constraints
    int get_num_active_constraints() const {return numActiveConstraints_;}
    // Set number of active constraints
    void set_num_active_constraints(int numActiveConstraints) {numActiveConstraints_ = numActiveConstraints;}
    // Get fixed constraint flag
    bool get_fixed_constraint_flag() const {return fixedConstraintsFlag_;}
    // Set fixed constraint flag
    void set_fixed_constraint_flag(bool flag) {fixedConstraintsFlag_ = flag;}

protected:
    // Number of vertices in the system
    int numVertices_;
    // Number of active constraints
    int numActiveConstraints_ = 0; 
    // Bool that indicates if constraints are fixed 
    bool fixedConstraintsFlag_ = false;
};

#endif