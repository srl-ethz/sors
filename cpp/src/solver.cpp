#include "solver.h"
#include "benchmark.h"

template<int vertexDim, int elementDim>
VectorXd Solver<vertexDim, elementDim>::step (VectorXd& prevSolution, Energy<vertexDim, elementDim>& systemEnergy, Params& act, double dt, int substeps) const {
    VectorXd solution = prevSolution;

    if (dt == 0.0)
        dt = systemEnergy.settings_.dt;

    if (substeps == 0)
        substeps = systemEnergy.settings_.substeps;

    bool CFLtimesteppingFlag = systemEnergy.settings_.CFLtimeSteppingFlag;
    std::string solverMethod = systemEnergy.settings_.solverMethod;

    if (CFLtimesteppingFlag) {
        // calculate current time step based on CFL condition (courrant constant = 1)
        double dtCFL = this->calculate_CFL_timestep(solution, systemEnergy, dt, 1);
        substeps = std::ceil(dt/dtCFL); 
    }

    double dtSubstep = dt/substeps;
    if (systemEnergy.settings_.verbose) {
        if (substeps == 1) std::cout << bcolors.OKCYAN << " (1 substep with dt = " << dtSubstep << ")" << bcolors.ENDC << std::endl;
        else std::cout << bcolors.OKCYAN << " (" << substeps << " substeps with dt = " << dtSubstep << ")" << bcolors.ENDC << std::endl;
    }
    for (int j = 0; j < substeps; j++) {
        if (solverMethod == "minimize_newton") {
            solution = this->minimize_newton(solution, systemEnergy, dtSubstep, act);
        } else if (solverMethod == "minimize_SQP") {
            solution = this->minimize_SQP(solution, systemEnergy, dtSubstep, act);
        } else if (solverMethod == "minimize_newton_linesearch") {
            solution = this->minimize_newton_linesearch(solution, systemEnergy, dtSubstep, act);
        } else if (solverMethod == "minimize_frozenhessian") {
            solution = this->minimize_frozenhessian(solution, systemEnergy, dtSubstep, act);
        } else {
            std::cout << bcolors.FAIL << "Solver method not recognized" << bcolors.ENDC << std::endl;
            throw std::runtime_error("Solver method not recognized");
        }
        systemEnergy.update_state(solution, dtSubstep);
    }
    return solution;
}


template<int vertexDim, int elementDim>
VectorXd Solver<vertexDim, elementDim>::step (VectorXd& solution, Energy<vertexDim, elementDim>& systemEnergy, double dt, int substeps) const {
    // Default step function without actuation
    Params act;
    return this->step(solution, systemEnergy, act, dt, substeps);
}


template<int vertexDim, int elementDim>
VectorXd Solver<vertexDim, elementDim>::minimize_newton (VectorXd& initialGuess, Energy<vertexDim, elementDim>& systemEnergy, double dt, Params& act) const {

    // Minimize the energy using Newton's method
    VectorXd x = initialGuess;
    SparseMatrix<double> hessian;
    VectorXd dx;
    VectorXd gradient;
    bool converged = false;

    SimplicialLDLT<SparseMatrix<double>> solver;

    for (int i = 0; i < MAX_ITER; i++) {
        // Compute the gradient and hessian
        // The scopes are used so that the profiler measures the time taken by each function separately (destructor called at end of scope and time measured)
        {
            PROFILE_SCOPE("Newton::compute_gradient");
            gradient = systemEnergy.compute_gradient(x, dt, act);
        }
        {
            PROFILE_SCOPE("Newton::compute_hessian");
            hessian = systemEnergy.compute_hessian(x, dt, act);
        }
        {
            PROFILE_SCOPE("Newton::solve");
            // Solve the linear system
            // LDLT works for both positive and negative semi-definite matrices, if we know the matrix is positive definite, we can use LLT
            solver.compute(hessian);
            // VectorXd dx = hessian.llt().solve(-gradient);
            dx = solver.solve(-gradient);
        }

        // Update the solution
        x += dx;
        // Check for convergence
        if (dx.hasNaN()) {
            std::cout << "Newton's method diverged" << std::endl;
            break;
        }
        else if (dx.norm() < DX_TOL) {
            converged = true;
            // std::cout << bcolors.OKGREEN << "Newton's method converged in " << i << " iterations" << bcolors.ENDC << std::endl;
            break;
        }
    }
    if (!converged) {
        std::cout << bcolors.FAIL << "Newton's method did not converge" << bcolors.ENDC << std::endl;
        throw std::runtime_error("Newton's method did not converge");
    }
    return x;
}


template<int vertexDim, int elementDim>
VectorXd Solver<vertexDim, elementDim>::minimize_SQP (VectorXd& initialGuess, Energy<vertexDim, elementDim>& systemEnergy, double dt, Params& act) const {

    VectorXd x = initialGuess;
    int numDOFs = x.size();
    SparseMatrix<double> hessian;

    // OSQP solver setup
    OSQPInt exitflag = 0;
    OSQPSettings *settings = OSQPSettings_new();
    if (settings) {
        osqp_set_default_settings(settings);
        settings->verbose = false;  // Disable solver output
        settings->scaling = 0;
        settings->max_iter = MAX_ITER_QP; // Default is 4000, but we rather have a lower number of QP iterations with more SQP steps
    }
    // Update the active constraints in the system energy
    bool activeConstraintsUpdated = systemEnergy.update_active_constraints(x);
    assert(activeConstraintsUpdated && "Error in updating active constraints in the system energy. Please check your implementation.");

    bool converged = false;
    for (int i = 0; i < MAX_ITER; i++) {
        // Compute the gradient and hessian
        VectorXd gradient = systemEnergy.compute_gradient(x, dt, act);
        hessian = systemEnergy.compute_hessian(x, dt, act);

        // CSC indices constructed and deleted inside of loop
        OSQPFloat* hessianCscValues; OSQPInt* hessianCscColIdx; OSQPInt* hessianCscRowIdx;
        OSQPFloat* constraintCscValues; OSQPInt* constraintCscColIdx; OSQPInt* constraintCscRowIdx;
        // Create CSC hessian (pass true for upperTriangular since hessian is symmetric)
        std::tie(hessianCscValues, hessianCscColIdx, hessianCscRowIdx) = utils::convert_eigen_to_osqpCsc(hessian, true);
        // Constraints: build lower and upper bounds vectors (size m)
        auto [lowerBounds, upperBounds] = systemEnergy.compute_constraint_bounds(x, dt);
        int m = lowerBounds.size();
        // Constraints: build constraint matrix A (m x 3n) for OSQP
        SparseMatrix<double> constraintMatrix = systemEnergy.compute_constraint_matrix_qp(x, dt);
        std::tie(constraintCscValues, constraintCscColIdx, constraintCscRowIdx) = utils::convert_eigen_to_osqpCsc(constraintMatrix);
        // Setup CSC matrices for OSQP
        OSQPCscMatrix* hessianCSC = OSQPCscMatrix_new(
            static_cast<OSQPInt>(hessian.rows()),
            static_cast<OSQPInt>(hessian.cols()),
            static_cast<OSQPInt>(hessian.nonZeros()),
            hessianCscValues, hessianCscColIdx, hessianCscRowIdx);
        OSQPCscMatrix* constraintMatrixCSC = OSQPCscMatrix_new(
            static_cast<OSQPInt>(constraintMatrix.rows()),
            static_cast<OSQPInt>(constraintMatrix.cols()),
            static_cast<OSQPInt>(constraintMatrix.nonZeros()),
            constraintCscValues, constraintCscColIdx, constraintCscRowIdx);

        // std::cout << bcolors.OKCYAN << "SQP iteration " << i << " with #constraints = " << m << " and #DoF = " << numDOFs << bcolors.ENDC << std::endl;

        // Setup solver
        OSQPSolver *solver;
        
        exitflag = osqp_setup(&solver, hessianCSC, gradient.data(), constraintMatrixCSC, lowerBounds.data(), upperBounds.data(), m, numDOFs, settings);
        if (!exitflag) exitflag = osqp_solve(solver);
        if (exitflag != 0) {
            std::cout << bcolors.FAIL << "OSQP solver failed with exitflag " << exitflag << bcolors.ENDC << std::endl;
            throw std::runtime_error("OSQP solver failed");
        }
     
        // update solution
        VectorXd dx = Eigen::Map<VectorXd>(solver->solution->x, numDOFs);
        x += dx;

        // Cleanup memory used by OSQP
        osqp_cleanup(solver);
        delete[] hessianCscValues; delete[] hessianCscColIdx; delete[] hessianCscRowIdx;
        delete[] constraintCscValues; delete[] constraintCscColIdx; delete[] constraintCscRowIdx;

        // Check for convergence
        if (dx.hasNaN()) {
            std::cout << "SQP method diverged" << std::endl;
            break;
        }
        else if (dx.norm() < DX_TOL) {
            converged = true;
            break;
        }

    }
    if (!converged) {
        std::cout << bcolors.FAIL << "SQP method did not converge" << bcolors.ENDC << std::endl;
        throw std::runtime_error("SQP method did not converge");
    }
    // Cleanup
    OSQPSettings_free(settings);

    return x;
}


template<int vertexDim, int elementDim>
VectorXd Solver<vertexDim, elementDim>::minimize_newton_linesearch (VectorXd& initialGuess, Energy<vertexDim, elementDim>& systemEnergy, double dt, Params& act) const {
    // Minimize the energy using Newton's method and backtracking line search
    VectorXd x = initialGuess;
    VectorXd xPrev = x;
    SparseMatrix<double> hessian;

    bool converged = false;
    const double ALPHA_INIT = 1.0; // Initial step size
    const double BETA = 0.5; // Reduction factor for step size
    const double C = 0.1; // Armijo condition constant

    for (int i = 0; i < MAX_ITER; i++) {
        // Compute the gradient and hessian
        VectorXd gradient = systemEnergy.compute_gradient(x, dt, act);
        hessian = systemEnergy.compute_hessian(x, dt, act);

        // Solve the linear system
        // LDLT works for both positive and negative semi-definite matrices, if we know the matrix is positive definite, we can use LLT
        SimplicialLDLT<SparseMatrix<double>> solver;
        solver.compute(hessian);

        // VectorXd dx = hessian.llt().solve(-gradient);
        VectorXd dx = solver.solve(-gradient); // Compute Newton direction

        // Line search initialization
        double alpha = ALPHA_INIT;

        // Backtracking line search
        while (true) {
            std::cout << "alpha: " << alpha << std::endl;
            VectorXd x_new = x + alpha*dx;
            double energy_new = systemEnergy.compute_energy(x_new, dt, act);
            double energy = systemEnergy.compute_energy(x, dt, act);
            double energy_expected = energy + C*alpha*gradient.dot(dx);
            if (energy_new <= energy_expected) {
                x = x_new;
                break;
            }
            alpha *= BETA;
        }

        // Update the solution
        dx = x - xPrev;
        xPrev = x;

        //std::cout << "dx norm: " << dx.norm() << std::endl;
        std::cout << "energy: " << systemEnergy.compute_energy(x, dt, act) << std::endl << "----------------" << std::endl;

        // Check for convergence
        if (dx.hasNaN()) {
            std::cout << "Newton's method diverged" << std::endl;
            break;
        }
        else if (dx.norm() < DX_TOL) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        std::cout << bcolors.FAIL << "Newton's method did not converge" << bcolors.ENDC << std::endl;
        throw std::runtime_error("Newton's method did not converge");
    }
    return x;
}


template<int vertexDim, int elementDim>
VectorXd Solver<vertexDim, elementDim>::minimize_frozenhessian (VectorXd& initialGuess, Energy<vertexDim, elementDim>& systemEnergy, double dt, Params& act) const {
    // Minimize the energy using Newton's method but freeze Hessian after first compute
    VectorXd x = initialGuess;

    SparseMatrix<double> hessian = systemEnergy.compute_hessian(x, dt, act);
    // Solve the linear system
    // LDLT works for both positive and negative semi-definite matrices, if we know the matrix is positive definite, we can use LLT
    SimplicialLDLT<SparseMatrix<double>> solver;
    solver.compute(hessian);

    bool converged = false;
    for (int i = 0; i < MAX_ITER; i++) {
        // Compute the gradient and hessian
        VectorXd gradient = systemEnergy.compute_gradient(x, dt, act);
        // VectorXd dx = hessian.llt().solve(-gradient);
        VectorXd dx = solver.solve(-gradient);
        // Update the solution
        x += dx;
        // Check for convergence
        if (dx.norm() < DX_TOL) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        std::cout << bcolors.FAIL << "Frozen Newton's method did not converge" << bcolors.ENDC << std::endl;
    }
    return x;
}


template<int vertexDim, int elementDim>
double Solver<vertexDim, elementDim>::calculate_CFL_timestep (VectorXd& solution, Energy<vertexDim, elementDim>& systemEnergy, double dt, double cflConstant) const {
    // Calculate the Courant-Friedrichs-Lewy condition for the system (simple 1D case for now)

    // Calculate maximal velocity (maximum absolute value from systemEnergy.vPrev_)
    double maxVelocity = systemEnergy.get_vPrev().cwiseAbs().maxCoeff();
    // Calculate characteristic length (in fluid usually miniumum grid size, hence we take the minimum edge length, loop through all elements)
    double minimalEdgeLength = -1.0;
    for (int i = 0; i < systemEnergy.eleIdx_.rows(); i++) {
        // Get the element vertex indices
        VectorXi ele = systemEnergy.eleIdx_.row(i);
        // Get the element vertices
        Matrix<double, elementDim, vertexDim> vertices;
        for (int j = 0; j < elementDim; j++) {
            for (int k = 0; k < vertexDim; k++) {
                vertices(j, k) = solution(vertexDim * ele(j) + k);
            }
        }
        // Calculate minimal edge length of this element
        double edgeLength = systemEnergy.elements_[i]->compute_minimal_edge_length(vertices);
        if (minimalEdgeLength < 0 || edgeLength < minimalEdgeLength) {
            minimalEdgeLength = edgeLength;
        }
    }

    // Calculate CFL condition
    double dtCFL = cflConstant * minimalEdgeLength / maxVelocity;
    if (dtCFL < dt) {
        return dtCFL;
    } else {
        return dt;
    }
}

template class Solver<3, 4>;
template class Solver<3, 8>;

