#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "utils.h"
#include "simulationSettings.h"
#include "params.h"
#include "io.h"

/**
 * @class TimeIntegrator
 * @brief Inertial energy assembly for implicit time integration.
 *
 * Builds the objective used by the solver by combining internal energy with inertial terms
 * (kinetic + gravity + Rayleigh-type damping) for Backward Euler and Crank–Nicolson.
 */
class TimeIntegrator {
public: 
    TimeIntegrator() {};

    // Constructor with time stepping method and previous position
    TimeIntegrator(const std::string method, const VectorXd& qPrev) {
        this->timeSteppingScheme_ = method;
        this->qPrev_ = qPrev;
        this->vPrev_ = VectorXd::Zero(qPrev.size());
        this->gPrev_ = VectorXd::Zero(qPrev.size());
        this->gPrevTmp_ = VectorXd::Zero(qPrev.size());
    };

    // Constructor with simulation settings and previous position
    TimeIntegrator(const SimulationSettings settings, const VectorXd& qPrev) {
        this->timeSteppingScheme_ = settings.timeSteppingScheme;
        this->qPrev_ = qPrev;
        this->vPrev_ = VectorXd::Zero(qPrev.size());
        this->gPrev_ = VectorXd::Zero(qPrev.size());
        this->gPrevTmp_ = VectorXd::Zero(qPrev.size());
    };

    /**
     * @brief Assemble total energy for implicit time integration.
     *
     * Combines internal energy with inertial contributions from kinetic energy,
     * gravity, and damping according to the selected time-stepping scheme.
     *
     * @param q                 Current generalized positions.
     * @param mass              Per-DOF mass vector.
     * @param gravAcceleration  Constant gravitational acceleration vector.
     * @param dampingAlpha      Rayleigh damping coefficient.
     * @param dt                Time step size.
     * @param energyInt         Internal (elastic + constraint) energy.
     * @return                  Total scalar energy at the current time step.
     */
    double integrate_energy (const VectorXd& q, const VectorXd& mass, const VectorXd& gravAcceleration, double dampingAlpha, double dt, double energyInt) const;

    /**
     * @brief Assemble total energy gradient for implicit time integration.
     *
     * Adds inertial force contributions (mass, gravity, damping) to the gradient
     * of the internal energy, using stored history where required by the scheme.
     * Takes gradient of internal energy as last parameter compared to integrate_energy.
     */
    VectorXd integrate_gradient (const VectorXd& q, const VectorXd& mass, const VectorXd& gravAcceleration, double dampingAlpha, double dt, const VectorXd& gradientInt);
    
    /**
     * @brief Assemble total Hessian for implicit time integration.
     *
     * Forms the second-order energy approximation by combining the internal
     * Hessian with diagonal inertial and damping terms implied by the time
     * integration scheme. Takes Hessian of internal energy as last parameter
     * compared to integrate_energy.
     */
    SparseMatrix<double> integrate_hessian (const VectorXd& q, const VectorXd& mass, const VectorXd& gravAcceleration, double dampingAlpha, double dt, const SparseMatrix<double>& hessianInt) const;

    /**
     * @brief Update stored state after a successful time step.
     *
     * Updates previous positions, velocities, and cached internal gradients
     * required for the next integration step.
     */
    void update_state (VectorXd& q, double dt) {
        this->vPrev_ = compute_velocity(q, dt);
        this->qPrev_ = q;
        this->gPrev_ = this->gPrevTmp_; 
    };

    // Compute velocity based on time stepping scheme
    VectorXd compute_velocity (VectorXd& q, double dt) const {
        // Compute the velocity of the system based on the current vertex positions q.
        if (this->timeSteppingScheme_ == "backward_euler") {
            VectorXd velocity = (q - qPrev_) / dt;
            return velocity;
        } else if (this->timeSteppingScheme_ == "crank_nicolson") {
            VectorXd velocity = 2 * (q - qPrev_) / dt - vPrev_;
            return velocity;
        } else {
            std::cerr << bcolors.FAIL << "Unknown time stepping scheme: " << this->timeSteppingScheme_ << bcolors.ENDC << std::endl;
            assert(false);
        }
    }

    double get_dvdx (double dt) const {
        if (this->timeSteppingScheme_ == "backward_euler") {
            return 1 / dt;
        }
        else if (this->timeSteppingScheme_ == "crank_nicolson") {
            return 2 / dt;
        }
        else {
            std::cerr << bcolors.FAIL << "Unknown time stepping scheme: " << this->timeSteppingScheme_ << bcolors.ENDC << std::endl;
            assert(false);
        }
    }

    // Setters and getters 
    void set_initial_deformation (MatrixXd& vertices) {qPrev_ = vertices.transpose().reshaped();}
    void set_initial_velocity (MatrixXd& velocities) {vPrev_ = velocities.transpose().reshaped();}
    VectorXd get_qPrev () const {return this->qPrev_;}
    VectorXd get_vPrev () const {return this->vPrev_;}

    // Member variables
    std::string timeSteppingScheme_;
    
private:
    // Previous position
    VectorXd qPrev_; 
    // Previous velocity
    VectorXd vPrev_;
    // Gradient of internal energy at previous time step
    VectorXd gPrev_; 
    // Temporary storage for gradient of internal energy at current time step
    VectorXd gPrevTmp_; 
};

#endif
