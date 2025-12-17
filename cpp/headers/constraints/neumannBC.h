#ifndef NEUMANNBC_H
#define NEUMANNBC_H

#include "constraint.h"

/**
 * @class NeumannBC
 * @brief Equality constraint enforcing Neumann-type boundary conditions on vertex DOFs.
 *
 * Inherits from Constraint. For each active degree of freedom, the constraint enforces
 * a discrete velocity condition of the form (x - x_prev)/dt = v_target, where v_target
 * is given by the Neumann boundary condition value.
 */
template<int vertexDim, int elementDim>
class NeumannBC : public Constraint<vertexDim, elementDim> {

public:
    /**
     * @brief Construct a Neumann boundary condition constraint.
     *
     * Initializes per-DOF masks and target velocities. Each entry in the mask activates
     * or deactivates a constraint on the corresponding scalar DOF. All active constraints
     * are fixed for the entire simulation.
     *
     * @param numVertices     Total number of vertices in the system.
     * @param neumannBCmask   Mask of size 3 * numVertices; 1 marks an active DOF, 0 an inactive one.
     * @param neumannBCvalue  Target velocities of size 3 * numVertices corresponding to each DOF.
     */
    NeumannBC(int numVertices, VectorXi neumannBCmask, VectorXd neumannBCvalue)
    : Constraint<vertexDim, elementDim>(numVertices), 
      neumannBCmask_(neumannBCmask), 
      neumannBCvalue_(neumannBCvalue) {
        // Check if the size of the mask and value vector is correct
        if (neumannBCmask.size() != 3 * static_cast<Eigen::Index>(numVertices) || neumannBCvalue_.size() != 3 * static_cast<Eigen::Index>(numVertices)) {
            std::cerr << bcolors.FAIL << "Error: Neumann boundary condition mask and value vector must have size 3 * numVertices." << bcolors.ENDC << std::endl;
            assert(neumannBCmask.size() == static_cast<Eigen::Index>(numVertices) && neumannBCvalue_.size() == static_cast<Eigen::Index>(numVertices));
        }
        // Set the number of active constraints
        this->set_num_active_constraints(neumannBCmask_.cast<bool>().count());
        // Set the fixed constraint flag to true, since Neumann boundary conditions are always fixed
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
        return "equality";
    }

    // Returns the name of the constraint
    std::string get_constraint_name() const override {
        return "neumannBC";
    }

    // Returns the BC mask
    VectorXi get_neumannBCmask() const {
        return neumannBCmask_;
    }
    
    // Returns the BC value
    VectorXd get_neumannBCvalue() const {
        return neumannBCvalue_;
    }

protected:
    // Neumann boundary condition mask (size 3n)
    VectorXi neumannBCmask_;
    // Neumann boundary condition value (size 3n)
    VectorXd neumannBCvalue_;
};

#endif