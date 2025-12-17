#ifndef SOLVER_H
#define SOLVER_H

#include "energy.h"
#include "simulationSettings.h"
#include "utils.h"
#include "osqp.h"

/**
 * @class Solver
 * @brief High-level nonlinear solver for implicit time-stepping.
 *
 * The Solver class wraps several energy-minimization strategies for advancing
 * an elastic system in time. Given an Energy instance and the current state,
 * it drives:
 * - unconstrained Newton minimization
 * - constrained SQP minimization (via OSQP)
 * - Newton with backtracking line search
 * - Newton with a frozen Hessian approximation
 *
 * It also supports optional CFL-based adaptive substepping and basic Hessian
 * conditioning diagnostics. The template parameters @p vertexDim and
 * @p elementDim encode the spatial dimension (e.g. 3) and number of vertices
 * per element (e.g. 4 for tets, 8 for hexes).
 */
template<int vertexDim, int elementDim>
class Solver {
public:
    Solver() {};
    // Following constructor used in Pybind to determine mesh type.
    Solver(SimulationSettings settings) : settings_(settings) {};
    SimulationSettings settings_;

    /**
     * @brief Advance the solution by one time step with actuation.
     *
     * This is the main driver for a time step. It:
     * - chooses the effective time step size @p dt (from argument or Energy settings)
     * - decides on substepping, either fixed (@p substeps) or CFL-based
     * - calls the selected nonlinear solver (Newton, SQP, etc.) for each substep
     * - updates the internal state of @p systemEnergy after each substep
     *
     * @param solution  Current state vector, used as initial guess and updated in-place.
     * @param systemEnergy Energy object providing energy, gradient, Hessian, and constraints.
     * @param act       Actuation parameters (e.g. pressure, muscle activation).
     * @param dt        Nominal time step; if zero, uses @c systemEnergy.settings_.dt.
     * @param substeps  Number of explicit substeps; if zero, uses settings or CFL logic.
     * @return Updated solution after completing the step.
     */
    VectorXd step (VectorXd& solution, Energy<vertexDim, elementDim>& systemEnergy, Params& act, double dt=0.0, int substeps=0) const;   
    
    // Convenience step overload without actuation
    VectorXd step (VectorXd& solution, Energy<vertexDim, elementDim>& systemEnergy, double dt=0.0, int substeps=0) const; 

    /**
     * @brief Unconstrained Newton minimization of the energy.
     *
     * Performs a fixed-number-of-steps Newton solve (up to MAX_ITER) using the
     * Energy-provided gradient and Hessian, with a sparse LDLT factorization.
     */
    VectorXd minimize_newton (VectorXd& initialGuess, Energy<vertexDim, elementDim>& systemEnergy, double dt, Params& act) const;
    
    /**
     * @brief Sequential Quadratic Programming (SQP) minimization with OSQP.
     *
     * Solves a sequence of quadratic subproblems with inequality/equality
     * constraints.
     */
    VectorXd minimize_SQP (VectorXd& initialGuess, Energy<vertexDim, elementDim>& systemEnergy, double dt, Params& act) const;
    
    /** 
    * @brief Newton minimization with backtracking line search.
    *
    * Similar to @c minimize_newton, but replaces the full step update with an
    * Armijo-style backtracking line search to improve robustness for highly
    * nonlinear energies.
    */
    VectorXd minimize_newton_linesearch (VectorXd& initialGuess, Energy<vertexDim, elementDim>& systemEnergy, double dt, Params& act) const;
    
    /**
     * @brief Newton minimization with a frozen Hessian.
     *
     * Computes the Hessian once at the initial guess and reuses it for all
     * iterations, effectively performing a quasi-Newton method with a fixed
     * curvature approximation.
     */
    VectorXd minimize_frozenhessian (VectorXd& initialGuess, Energy<vertexDim, elementDim>& systemEnergy, double dt, Params& act) const;

    /**
     * @brief Compute a CFL-limited time step based on current velocities and mesh size.
     *
     * Estimates a stable time step based on a simple Courant-Friedrichs-Lewy
     * condition. 
     */
    double calculate_CFL_timestep (VectorXd& solution, Energy<vertexDim, elementDim>& systemEnergy, double dt, double cflConstant) const;
};

#endif