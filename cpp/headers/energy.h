#ifndef ENERGY_H
#define ENERGY_H

#include "utils.h"
#include "simulationSettings.h"
#include "params.h"
#include "io.h"
#include "integrator.h"
#include "element.h"
#include "tetrahedron.h"
#include "hexahedron.h"
#include "constraint.h"
#include "neumannBC.h"
#include "planeContact.h"
#include "plane3D.h"
#include "maxDistance.h"
#include "disk3D.h"
#include "diskContact.h"
#include "externalForce.h"
#include "pressureForce.h"
#include "vertexForce.h"

/**
 * @class Energy
 * @brief Global assembly class for element energies, constraints, and external forces.
 *
 * The Energy class manages the full physical system. It combines:
 * - finite elements (tetrahedra or hexahedra) and their material energy models
 * - global constraints (plane/disk contact, Neumann BCs, max-distance constraints, …)
 * - external forces (pressure, per-vertex forces, actuation-driven forces)
 * - the time integration scheme (velocities, accelerations, kinetic energy)
 *
 * It provides high-level functionality required by Newton/SQP solvers:
 * - compute_energy()        → total potential + kinetic energy
 * - compute_gradient()      → global force / energy gradient
 * - compute_hessian()       → sparse global Hessian of the energy
 * - compute_constraint_bounds(), compute_constraint_matrix_qp()
 * - update_state()          → advances time integrator and element states
 *
 * Template parameters:
 *  @tparam vertexDim  Spatial dimension of each vertex 
 *  @tparam elementDim Number of vertices per element (4 for tets, 8 for hexes)
 */
template <int vertexDim, int elementDim>
class Energy
{
public:
    Energy() {};
    /**
     * @brief Constructs the full global energy system for a given mesh.
     *
     * The constructor performs all system initialization:
     * - builds finite elements (tet/hex) from the undeformed mesh
     * - attaches the appropriate energy models to each element
     * - constructs constraints based on type strings and parameter maps
     * - constructs external forces (pressure, vertex forces, etc.)
     * - extracts surface groups and boundary vertices (for contact/pressure)
     * - assembles the lumped mass vector
     * - initializes the time integrator
     *
     * @param undeformedVertices    N×vertexDim undeformed vertex coordinates
     * @param eleIdx                M×elementDim connectivity matrix
     * @param settings              Simulation and solver settings
     * @param gravAcceleration      Gravity vector applied to all elements
     * @param dampingAlpha          Rayleigh damping alpha coefficient
     * @param elementEnergiesList   Per-element sets of active energy model names
     * @param elementParameterList  Per-element material/energy parameters
     * @param constraintTypesList   List of constraint type identifiers
     * @param constraintParameterList Parameter map for all constraints
     * @param forceTypesList        List of external force type identifiers
     * @param forceParameterList    Parameter map for all external forces
     */
    Energy(
        Matrix<double, -1, vertexDim>& undeformedVertices, 
        Matrix<int, -1, elementDim>& eleIdx,
        SimulationSettings settings,
        Vector<double, vertexDim> gravAcceleration={0.0, 0.0, 0.0},
        double dampingAlpha=0.0,
        std::vector<std::set<std::string>> elementEnergiesList={},
        std::vector<Params> elementParameterList={},
        std::vector<std::string> constraintTypesList={},
        Params constraintParameterList={},
        std::vector<std::string> forceTypesList={},
        Params forceParameterList={});

    /**
     * @brief Computes the total system energy at configuration x.
     *
     * The result includes:
     * - the sum of all element potential energies
     * - work performed by external forces (if present)
     * - kinetic energy contribution via the time integrator
     *
     * @param x          Current DOF vector
     * @param dt         Time step size
     * @param actuation  Actuation values for forces/energies
     *
     * @return Total potential + kinetic energy of the system
     */
    double compute_energy (VectorXd& x, double dt, Params& actuation) const;     
    
    /**
     * @brief Computes the global gradient (force vector) of the system energy.
     *
     * Assembles the global gradient from:
     * - element energy gradients
     * - external force contributions (subtracted for force balance)
     * - inertial terms introduced by the time integrator
     *
     * Behavior with constraints:
     * - Neumann BCs overwrite specific DOFs when using Newton’s method
     *
     * @param x          Current DOF vector
     * @param dt         Time step size
     * @param actuation  Actuation values for forces/energies
     *
     * @return Global gradient vector matching the size of x
     */
    VectorXd compute_gradient (VectorXd& x, double dt, Params& actuation);
    
    /**
     * @brief Computes the sparse Hessian matrix of the system energy.
     *
     * The Hessian includes:
     * - element Hessians (material + geometric stiffness)
     * - external force Hessians (when available)
     * - mass/time-integration contributions
     *
     * Constraint effects:
     * - Neumann BCs introduce diagonal regularization for constrained DOFs
     *
     * @param x          Current DOF vector
     * @param dt         Time step size
     * @param actuation  Actuation values for forces/energies
     *
     * @return Sparse global Hessian matrix of size (ndof × ndof)
     */
    SparseMatrix<double> compute_hessian (VectorXd& x, double dt, Params& actuation) const;  
  
    /**
     * @brief Computes the bound vectors (l, u) for all active constraints.
     *
     * Constraint interpretation:
     * - equality constraint g(x) = 0     → l = u = -g(x)
     * - inequality constraint g(x) ≥ 0   → l = -g(x), u = +∞
     *
     * The bounds are stacked in the same order as the constraint rows in the QP.
     *
     * @param x          Current DOF vector
     * @param dt         Time step size
     *
     * @return Pair (lowerBounds, upperBounds)
     */
    std::pair<VectorXd, VectorXd> compute_constraint_bounds (VectorXd& x, double dt) const;
    
    /**
     * @brief Assembles the stacked constraint Jacobian matrix A for all active constraints.
     *
     * Each active constraint contributes one row representing ∂gᵢ/∂x.
     * Rows are stacked in the same order as produce compute_constraint_bounds().
     *
     * @param x   Current DOF vector
     * @param dt  Time step size
     *
     * @return Sparse matrix A of size (#active constraints × ndof)
     */
    SparseMatrix<double> compute_constraint_matrix_qp (VectorXd& x, double dt) const;
    
    /**
     * @brief Updates the active set of every constraint in the system.
     *
     * For each constraint object:
     * - If not marked as fixed, its internal active set is recomputed.
     * - Returns false if any constraint reports an update failure.
     *
     * @param x   Current DOF vector
     * @return true if all constraints updated successfully, false otherwise
     */
    bool update_active_constraints(VectorXd& x) const;

    /**
     * @brief Updates time-integrator state and stateful energy models.
     *
     * Performs:
     * - time integrator update (positions, velocities)
     * - per-element state update 
     *
     * @param q   Current DOF vector
     * @param dt  Time step size
     */
    void update_state (VectorXd& q, double dt);

    // Compute elementwise energy for visualization (VisualizationOption is a list of energy names and/or "all" for total energy)
    std::vector<VectorXd> compute_elementwise_energy (VectorXd& q, double dt, Params& actuation, const std::vector<std::string>& VisualizationOption={"all"}) const;
    // Compute velocity of the system based on the current vertex positions q and given time stepping scheme
    VectorXd compute_velocity (VectorXd& q, double dt) const;
    // Compute acceleration of the system based on the current vertex positions q and given time stepping scheme
    VectorXd compute_acceleration (VectorXd& q, double dt) const;
    // Print physical properties of the system
    void verbose_print_physics_properties () const;
    
    // Setters and Getters
    void set_initial_deformation (MatrixXd& vertices) {this->timeIntegrator_.set_initial_deformation(vertices);}
    void set_initial_velocity (MatrixXd& velocities) {this->timeIntegrator_.set_initial_velocity(velocities);}
    VectorXd get_vPrev () const {return this->timeIntegrator_.get_vPrev();}

    // Vector of elements, each element is a Tetrahedron or a Hexahedron object
    std::vector<std::unique_ptr<Element<vertexDim, elementDim>>> elements_; 
    // Vector of constraints, each constraint is a Constraint object
    std::vector<std::unique_ptr<Constraint<vertexDim,elementDim>>> constraints_; 
    // Vector of external forces, each force is an ExternalForce object
    std::vector<std::unique_ptr<ExternalForce<vertexDim,elementDim>>> externalForces_; 
    // Element indices, each row is an element, each column is a vertex index
    Matrix<int, -1, elementDim> eleIdx_; 
    // Surface groups, each group is a vector of vertices, each vertex is a vector of indices
    std::vector<std::vector<std::vector<int>>> surfaceGroups_; 
    // Face element groups, each group is a vector of element indices, each element is a vector of indices
    std::vector<std::vector<int>> faceEleGroups_; 
    // Set of surface vertices indices, used for visualization and constraints
    std::set<int> surfaceVerticesIdx_; 
    // Simulation settings
    SimulationSettings settings_; 
    // Gravitational acceleration
    Vector<double, vertexDim> gravAcceleration_; 
    // Damping alpha coefficient
    double dampingAlpha_;

private:
    // Number of vertices and elements
    int numVertices_;
    int numElements_;
    // Undeformed vertices
    Matrix<double, -1, vertexDim> undeformedVertices_;
    // List of element energies
    std::vector<std::set<std::string>> elementEnergiesList_; 
    // List of constraint types
    std::vector<std::string> constraintTypesList_;
    // List of force types
    std::vector<std::string> forceTypesList_; 
    // Flag to indicate if external vertex forces are set
    bool externalForcesFlag_; 
    // Flag to indicate if local energy update is needed
    bool localEnergyUpdateFlag_ = false; 
    // Mask for Neumann boundary conditions
    VectorXi neumannBCmask_; 
    // Values for Neumann boundary conditions
    VectorXd neumannBCvalue_; 
    // Mass per vertex
    VectorXd mass_;
    // Time integrator
    TimeIntegrator timeIntegrator_; 
};

#endif
