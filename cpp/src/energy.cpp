#include "energy.h"

template<int vertexDim, int elementDim>
Energy<vertexDim, elementDim>::Energy (
    Matrix<double, -1, vertexDim>& undeformedVertices, 
    Matrix<int, -1, elementDim>& eleIdx,
    SimulationSettings settings,
    Vector<double, vertexDim> gravAcceleration,
    double dampingAlpha,
    std::vector<std::set<std::string>> elementEnergiesList,
    std::vector<Params> elementParameterList,
    std::vector<std::string> constraintTypesList,
    Params constraintParameterList,
    std::vector<std::string> forceTypesList,
    Params forceParameterList) { 
    this->eleIdx_ = eleIdx;
    this->numElements_ = eleIdx.rows();
    this->undeformedVertices_ = undeformedVertices;
    this->numVertices_ = undeformedVertices.rows();
    this->neumannBCmask_ = VectorXi::Zero(vertexDim * numVertices_); 
    this->neumannBCvalue_ = VectorXd::Zero(vertexDim * numVertices_);
    this->gravAcceleration_ = gravAcceleration;
    this->dampingAlpha_ = dampingAlpha;

    // Element energies
    this->elementEnergiesList_ = elementEnergiesList;
    // Constraint types
    this->constraintTypesList_ = constraintTypesList;

    // Set flag if we need surface extraction for constraints/forces
    bool surfaceExtractionNeeded = false;
    for (unsigned int i = 0; i < constraintTypesList_.size(); i++) {
        if (constraintTypesList_[i] == "planeContact" || constraintTypesList_[i] == "diskContact") {
            surfaceExtractionNeeded = true;
        }
    }
    for (unsigned int i = 0; i < forceTypesList.size(); i++) {
        if (forceTypesList[i] == "pressure") {
            surfaceExtractionNeeded = true;
        }
    }

    // Store surface groups, face element groups, and surface vertices
    if (surfaceExtractionNeeded) {
        auto [surfaceGroups, faceEleGroups] =
            utils::extract_surfaces<vertexDim, elementDim>(undeformedVertices, eleIdx);
        std::set<int> surfaceVerticesIdx;
        for (const auto& group : surfaceGroups) {
            for (const auto& face : group) {
                surfaceVerticesIdx.insert(face.begin(), face.end());
            }
        }
        this->surfaceGroups_      = std::move(surfaceGroups);
        this->faceEleGroups_      = std::move(faceEleGroups);
        this->surfaceVerticesIdx_ = std::move(surfaceVerticesIdx);
    }
    
    // Initialize constraints
    constraints_ = std::vector<std::unique_ptr<Constraint<vertexDim, elementDim>>>(constraintTypesList_.size());
    for (unsigned int i = 0; i < constraintTypesList_.size(); i++) {
        if (constraintTypesList_[i] == "neumannBC") {
            // Additional input parameters: neumannBCmask, neumannBCvalue
            VectorXi neumannBCmask = constraintParameterList.get_value("neumannBCmask").reshaped<RowMajor>().cast<int>();
            VectorXd neumannBCvalue = constraintParameterList.get_value("neumannBCvalue").reshaped<RowMajor>();
            constraints_[i] = std::make_unique<NeumannBC<vertexDim, elementDim>>(numVertices_, neumannBCmask, neumannBCvalue);
            this->neumannBCmask_ = neumannBCmask; // Set member variables for easy access
            this->neumannBCvalue_ = neumannBCvalue;
        }
        else if (constraintTypesList_[i] == "planeContact") {
            // Additional input parameters: list of planes (positionVectorListPlanes, normalVectorListPlanes), surface vertices indices
            VectorXd positionVectorList = constraintParameterList.get_value("positionVectorListPlanes").reshaped<RowMajor>();
            VectorXd normalVectorList = constraintParameterList.get_value("normalVectorListPlanes").reshaped<RowMajor>();
            assert(positionVectorList.size() == normalVectorList.size() && positionVectorList.size() % 3 == 0 && normalVectorList.size() % 3 == 0);
            std::vector<Plane3D> planes; 
            for (Eigen::Index j = 0; j < positionVectorList.size() / 3; j++) {
                Vector3d position(positionVectorList(3 * j), positionVectorList(3 * j + 1), positionVectorList(3 * j + 2));
                Vector3d normal(normalVectorList(3 * j), normalVectorList(3 * j + 1), normalVectorList(3 * j + 2));
                Plane3D plane(position, normal);
                planes.push_back(plane);
            }
            VectorXi surfaceVerticesIdx = constraintParameterList.get_value("surfaceVerticesIdx").reshaped<RowMajor>().cast<int>(); 
            assert(surfaceVerticesIdx.size() > 0 && "Surface vertices indices cannot be empty for PlaneContact constraint.");
            constraints_[i] = std::make_unique<PlaneContact<vertexDim, elementDim>>(numVertices_, planes, surfaceVerticesIdx); 
        } 
        else if (constraintTypesList_[i] == "diskContact") {
            // Additional input parameters: list of disks (diskCenterVectorList, diskNormalVectorList, diskRadiusList), surface vertices indices
            VectorXd diskCenterVectorList = constraintParameterList.get_value("diskCenterVectorList").reshaped<RowMajor>();
            VectorXd diskNormalVectorList = constraintParameterList.get_value("diskNormalVectorList").reshaped<RowMajor>();
            VectorXd diskRadiusList = constraintParameterList.get_value("diskRadiusList").reshaped<RowMajor>();
            assert(diskCenterVectorList.size() == diskNormalVectorList.size()
            && diskCenterVectorList.size() % 3 == 0
            && diskNormalVectorList.size() % 3 == 0
            && "DiskContact: centers/normals must be same length and multiples of 3.");
            const Eigen::Index numDisks = diskCenterVectorList.size() / 3;
            assert(diskRadiusList.size() == numDisks && "DiskContact: radiusList size must match number of disks.");
            std::vector<Disk3D> disks;
            disks.reserve(static_cast<size_t>(numDisks));
            for (Eigen::Index j = 0; j < numDisks; ++j) {
                Vector3d center(diskCenterVectorList(3*j), diskCenterVectorList(3*j + 1), diskCenterVectorList(3*j + 2));
                Vector3d normal(diskNormalVectorList(3*j), diskNormalVectorList(3*j + 1), diskNormalVectorList(3*j + 2));
                double radius = diskRadiusList(j);
                Disk3D disk(center, normal, radius);
                disks.push_back(disk);
            }
            VectorXi surfaceVerticesIdx = constraintParameterList.get_value("surfaceVerticesIdx").reshaped<RowMajor>().cast<int>();
            assert(surfaceVerticesIdx.size() > 0 && "DiskContact: surfaceVerticesIdx cannot be empty.");
            constraints_[i] = std::make_unique<DiskContact<vertexDim, elementDim>>(numVertices_, disks, surfaceVerticesIdx);
        }
        else if (constraintTypesList_[i] == "maxDistance") {
            // Additional input parameters: maxDistanceVertex1Indices, maxDistanceVertex1Indices, maxDistanceList
            VectorXi vertex1Indices = constraintParameterList.get_value("maxDistanceVertex1Indices").reshaped<RowMajor>().cast<int>();
            VectorXi vertex2Indices = constraintParameterList.get_value("maxDistanceVertex2Indices").reshaped<RowMajor>().cast<int>();
            VectorXd maxDistanceList = constraintParameterList.get_value("maxDistanceList").reshaped<RowMajor>();
            assert(vertex1Indices.size() == vertex2Indices.size() && vertex1Indices.size() == maxDistanceList.size() && "MaxDistance constraint parameters must have the same size.");
            std::vector<std::tuple<int, int, double>> maxDistanceConstraintPairs;
            for (Eigen::Index j = 0; j < vertex1Indices.size(); j++) {
                maxDistanceConstraintPairs.emplace_back(vertex1Indices(j), vertex2Indices(j), maxDistanceList(j));
            }
            assert(maxDistanceConstraintPairs.size() > 0 && "MaxDistance constraint pair is empty. Remove this constraint.");
            constraints_[i] = std::make_unique<MaxDistance<vertexDim, elementDim>>(numVertices_, maxDistanceConstraintPairs);
        }
        else {
            std::cout << bcolors.FAIL << "Unknown constraint type: " << constraintTypesList_[i] << bcolors.ENDC << std::endl;
            exit(1);
        }
        // Add other constraint types here as needed
    }

    // Create vector of local elements
    this->elements_.resize(numElements_);
    for (int i = 0; i < numElements_; i++) {
        // Get the element vertex indices
        VectorXi ele = eleIdx_.row(i);
        // Disassemble into the element vertices
        Matrix<double, elementDim, vertexDim> vertices;
        for (int j = 0; j < elementDim; j++) {
            for (int k = 0; k < vertexDim; k++) {
                vertices(j, k) = undeformedVertices_(ele(j), k);
            }
        }
        // Create the element after checking what type of element: Tet or Hex
        if constexpr (vertexDim == 3 && elementDim == 4) {
            this->elements_[i] = std::make_unique<Tetrahedron>(elementParameterList[i], vertices, elementEnergiesList[i]);
        }
        else if constexpr (vertexDim == 3 && elementDim == 8) {
            this->elements_[i] = std::make_unique<Hexahedron>(elementParameterList[i], vertices, elementEnergiesList[i]);
        }
        else {
            // Not yet implemented
            assert(false);
        }
    }

    // External forces
    this->externalForcesFlag_ = (forceTypesList.size() > 0);
    this->forceTypesList_ = forceTypesList;
    externalForces_ = std::vector<std::unique_ptr<ExternalForce<vertexDim, elementDim>>>(forceTypesList_.size());
    for (unsigned int i = 0; i < forceTypesList_.size(); i++) {
        if (forceTypesList_[i] == "pressure") {
            // Create an instance of the PressureForce external force
            // Additional input parameters: surfaceVertexIdx, surfaceGroups
            MatrixXi surfaceVertexIdx = forceParameterList.get_value("surfaceVertexIdx").cast<int>();
            VectorXi surfaceGroups = forceParameterList.get_value("surfaceGroups").reshaped<RowMajor>().cast<int>();

            externalForces_[i] = std::make_unique<PressureForce<vertexDim, elementDim>>(surfaceVertexIdx, surfaceGroups);
        }
        else if (forceTypesList_[i] == "vertexForce") {
            externalForces_[i] = std::make_unique<VertexForce<vertexDim, elementDim>>();
        }
        else {
            std::cout << bcolors.FAIL << "Unknown force type: " << forceTypesList_[i] << bcolors.ENDC << std::endl;
            exit(1);
        }
    }

    // Build global mass matrix
    this->mass_ = VectorXd::Zero(numVertices_ * vertexDim);
    #pragma omp parallel for reduction(+:mass_)
    for (int i = 0; i < numElements_; i++) {
        // Get the element vertex indices
        VectorXi ele = eleIdx_.row(i);
        double elementMass = this->elements_[i]->compute_mass();
        // Assemble into the global mass vector
        for (int j = 0; j < elementDim; j++) {
            for (int k = 0; k < vertexDim; k++) {
                mass_(vertexDim * ele(j) + k) += elementMass / elementDim;
            }
        }
    }

    // Simulation settings
    this->settings_ = settings;
    
    // Set number of threads for OpenMP
    omp_set_dynamic(0);
    omp_set_num_threads(settings_.numThreads);
    // Initialize time integrator
    this->timeIntegrator_ = TimeIntegrator(settings, undeformedVertices.transpose().reshaped());

    // Print physics properties
    if (settings_.verbose) {
        this->verbose_print_physics_properties();
    }
}


template<int vertexDim, int elementDim>
double Energy<vertexDim, elementDim>::compute_energy (VectorXd& q, double dt, Params& actuation) const {
    // Compute the energy of the system based on the current vertex positions q.
    double energy = 0.0;
    // Compute the energy for each element by assembling the local contributions
    #pragma omp parallel for reduction(+:energy)
    for (int i = 0; i < numElements_; i++) {
        // Get the element vertex indices
        VectorXi ele = eleIdx_.row(i);
        // Disassemble into the element vertices
        Matrix<double, elementDim, vertexDim> vertices;
        for (int j = 0; j < elementDim; j++) {
            for (int k = 0; k < vertexDim; k++) {
                vertices(j, k) = q(vertexDim * ele(j) + k);
            }
        }
        // Compute the local element energy
        double elementEnergy = this->elements_[i]->compute_energy(vertices, actuation);
        // Add the element energy to the total global energy
        energy += elementEnergy;
    }
    // Compute energy contributions for each vertex individually due to external vertex forces
    if (externalForcesFlag_) {
        // Compute total external forces
        VectorXd externalForces = VectorXd::Zero(numVertices_ * vertexDim);
        for (std::size_t i = 0; i < this->externalForces_.size(); i++) {
            VectorXd act;
            if (this->externalForces_[i]->actuationFlag_) {
                act = actuation.get_value(this->externalForces_[i]->get_force_name()).transpose().reshaped();
            }
            else {
                act = VectorXd::Zero(0); // Default actuation if not specified
            }
            VectorXd externalForce = this->externalForces_[i]->compute_force(q, act);
            externalForces += externalForce;
        }
        energy += externalForces.dot(q-this->timeIntegrator_.get_qPrev()); // Work done by external forces
    }
    // Add kinetic energy through time integrator
    energy = this->timeIntegrator_.integrate_energy(q, this->mass_, this->gravAcceleration_, this->dampingAlpha_, dt, energy);
    return energy;
}


template<int vertexDim, int elementDim>
VectorXd Energy<vertexDim, elementDim>::compute_gradient (VectorXd& q, double dt, Params& actuation) {
    // Compute the gradient of the energy
    VectorXd gradient = VectorXd::Zero(q.size());
    // Compute the energy gradient by assembling the local contributions
    #pragma omp parallel for reduction(+:gradient)
    for (int i = 0; i < numElements_; i++) {
        // Get the element vertex indices
        VectorXi ele = eleIdx_.row(i);
        // Disassemble into the element vertices
        Matrix<double, elementDim, vertexDim> vertices;
        for (int j = 0; j < elementDim; j++) {
            for (int k = 0; k < vertexDim; k++) {
                vertices(j, k) = q(vertexDim * ele(j) + k);
            }
        }
        // Compute the local element energy gradient
        Vector<double, elementDim * vertexDim> elementGradient = this->elements_[i]->compute_gradient(vertices, actuation);
        // Add the element gradients to the total global energy gradient
        for (int j = 0; j < elementDim; j++)
        {
            for (int k = 0; k < vertexDim; k++)
            {
                // If no fixed or no neumann nodes (no boundary conditions) -> put in calculated gradient
                if (this->settings_.solverMethod == "minimize_SQP" || neumannBCmask_[vertexDim * ele(j) + k] == 0) {
                    gradient(vertexDim * ele(j) + k) += elementGradient(vertexDim * j + k);
                }
            }
        }
    }

    // Add external forces to the gradient
    for (std::size_t i = 0; i < this->externalForces_.size(); i++) {
        VectorXd act;
        if (this->externalForces_[i]->actuationFlag_) {
            act = actuation.get_value(this->externalForces_[i]->get_force_name()).transpose().reshaped();
        }
        else {
            act = VectorXd::Zero(0); // Default actuation if not specified
        }
        VectorXd externalForce = this->externalForces_[i]->compute_force(q, act);
        gradient -= externalForce;
    }

    // Apply Kinetic Energy for Force Balance
    gradient = this->timeIntegrator_.integrate_gradient(q, this->mass_, this->gravAcceleration_, this->dampingAlpha_, dt, gradient);

    // Add the neumann boundary conditions
    if (this->settings_.solverMethod == "minimize_newton" && neumannBCmask_.size() > 0) {
        VectorXd qPrev = this->timeIntegrator_.get_qPrev();
        for (int i = 0; i < numVertices_ * vertexDim; i++) {
            if (neumannBCmask_[i] == 1) {
                gradient(i) = -(qPrev(i) + dt * neumannBCvalue_[i] - q(i));
            }
        }
    }
    return gradient;
}


template<int vertexDim, int elementDim>
SparseMatrix<double> Energy<vertexDim, elementDim>::compute_hessian (VectorXd& q, double dt, Params& actuation) const {
    // Compute the hessian of the energy
    SparseMatrix<double> hessian(q.size(), q.size());
    std::vector<Triplet<double>> tripletList(numElements_*elementDim*vertexDim*elementDim*vertexDim); // Upper estimate, might be too large
    // Compute the energy hessian by assembling the local contributions
    #pragma omp parallel for
    for (int i = 0; i < numElements_; i++) {
        // Get the element vertex indices
        VectorXi ele = eleIdx_.row(i);
        // Disassemble into the element vertices
        Matrix<double, elementDim, vertexDim> vertices;
        for (int j = 0; j < elementDim; j++) {
            for (int k = 0; k < vertexDim; k++) {
                vertices(j, k) = q(vertexDim * ele(j) + k);
            }
        }
        // Compute the local element energy hessian (Note, we give a zero acceleration for now, as it is not needed for the hessian computation)
        Matrix<double, elementDim * vertexDim, elementDim * vertexDim> elementHessian = this->elements_[i]->compute_hessian(vertices, actuation);

        // TODO: Can be optimized with sparse matrices
        // Add the element hessians to the total global energy hessian
        for (int j = 0; j < elementDim; j++) {
            for (int k = 0; k < vertexDim; k++) {
                for (int m = 0; m < elementDim; m++) {
                    for (int l = 0; l < vertexDim; l++) {
                        if (this->settings_.solverMethod == "minimize_SQP" || (neumannBCmask_[vertexDim * ele(j) + k] == 0 && neumannBCmask_[vertexDim * ele(m) + l] == 0)) {
                            if (this->settings_.timeSteppingScheme == "backward_euler")
                                tripletList[i*elementDim*vertexDim*elementDim*vertexDim + j*vertexDim*elementDim*vertexDim + k*elementDim*vertexDim + m*vertexDim + l] \
                                    = Triplet<double>(vertexDim * ele(j) + k, vertexDim * ele(m) + l, 
                                    elementHessian(vertexDim * j + k, vertexDim * m + l));
                            else if (this->settings_.timeSteppingScheme == "crank_nicolson")
                                tripletList[i*elementDim*vertexDim*elementDim*vertexDim + j*vertexDim*elementDim*vertexDim + k*elementDim*vertexDim + m*vertexDim + l] \
                                = Triplet<double>(vertexDim * ele(j) + k, vertexDim * ele(m) + l,  elementHessian(vertexDim * j + k, vertexDim * m + l));
                        }
                    }
                }
            }
        }
    }

    // Add external forces to the hessian for Newton solve
    for (std::size_t i = 0; i < this->externalForces_.size(); i++) {
        VectorXd act;
        if (this->externalForces_[i]->actuationFlag_) {
            act = actuation.get_value(this->externalForces_[i]->get_force_name()).transpose().reshaped();
        }
        else {
            act = VectorXd::Zero(0); // Default actuation if not specified
        }

        std::vector<Triplet<double>> externalForceGradient = this->externalForces_[i]->compute_force_gradient(q, act);

        // Reserve new space in tripletList and append entries
        std::size_t prevSize = tripletList.size();
        tripletList.resize(prevSize + externalForceGradient.size());

        #pragma omp parallel for
        for (std::size_t j = 0; j < externalForceGradient.size(); j++) {
            Triplet<double> triplet = externalForceGradient[j];
            if (this->settings_.solverMethod == "minimize_newton" && (neumannBCmask_[triplet.row()] == 0 && neumannBCmask_[triplet.col()] == 0)) {
                tripletList[prevSize + j] = triplet;
            }
        }
    }
    
    hessian.setFromTriplets(tripletList.begin(), tripletList.end());
    // Time integration TODO: Might be inefficient since it's another O(N) loop.
    hessian = this->timeIntegrator_.integrate_hessian(q, this->mass_, this->gravAcceleration_, this->dampingAlpha_, dt, hessian);
    // Add the neumann boundary conditions
    if (this->settings_.solverMethod == "minimize_newton" && neumannBCmask_.size() > 0) {
        for (int i = 0; i < numVertices_ * vertexDim; i++) {
            if (neumannBCmask_[i] == 1) {
                // hessian.coeffRef(i, i) = 100.0/(dt*dt); // The value needs to roughly match the rest of the entries in the Hessian, otherwise the row of all zeros and a single one could be considered an all-zero row and cause the Hessian to be ill conditioned. Mainly the first Lame parameter that creates large Hessian entries.
                hessian.coeffRef(i, i) = 1.0; // Or zero the column as well above, and simply set 1 in the diagonal. This would cause the Newton step to be incorrect, but once converged it finds the correct equilibrium.
            }
        }
    }
    return hessian;
}


template<int vertexDim, int elementDim>
std::pair<VectorXd, VectorXd> Energy<vertexDim, elementDim>::compute_constraint_bounds (VectorXd& q, double dt) const {
    // Loop through all constraints first to find number of active constraints per constraint type
    VectorXi activeConstraints = VectorXi::Zero(constraints_.size()); // active constraints per constraint type
    for (std::size_t i = 0; i < constraints_.size(); i++) {
        // Compute the number of active constraints per constraint type
        activeConstraints(i) = constraints_[i]->get_num_active_constraints();
    }
    int numActiveConstraints = activeConstraints.sum(); // Total number of active constraints
    // Create the lower and upper bounds vectors
    VectorXd lowerBounds = VectorXd::Zero(numActiveConstraints);
    VectorXd upperBounds = VectorXd::Zero(numActiveConstraints);
    int offset = 0;
    for (std::size_t i = 0; i < constraints_.size(); i++) {
        // Active constraints of the current constraint type
        int numActive = activeConstraints(i);
        if (numActive == 0) continue; // skip if no active constraints
        // Compute the lower and upper bounds for each constraint
        std::string constraintType = constraints_[i]->get_constraint_type();
        VectorXd constraintVector = constraints_[i]->compute_constraint(q, this->timeIntegrator_.get_qPrev(), dt); // Computes the constraint vector g(x)
        assert(constraintVector.size() == numActive);
        if (constraintType == "equality") {
            // Equality constraint: g(x) == 0  ->  A dx = -g(x)  -> l = u = -g(x)
            lowerBounds.segment(offset, numActive) = -constraintVector;
            upperBounds.segment(offset, numActive) = -constraintVector;
        } else if (constraintType == "inequality") {
            // Inequality constraint: g(x) >= 0 -> A dx >= -g(x) -> l = -g(x), u = inf
            lowerBounds.segment(offset, numActive) = -constraintVector;
            upperBounds.segment(offset, numActive).setConstant(std::numeric_limits<double>::infinity());
        }
        offset += numActive; // Update offset for next constraint type
    }
    return std::make_pair(lowerBounds, upperBounds);
}
    

template<int vertexDim, int elementDim>
SparseMatrix<double> Energy<vertexDim, elementDim>::compute_constraint_matrix_qp (VectorXd& q, double dt) const {
    // Loop through all constraints first to find number of active constraints per constraint type
    VectorXi activeConstraints = VectorXi::Zero(constraints_.size()); // Active constraints per constraint type
    for (std::size_t i = 0; i < constraints_.size(); i++) {
        // Compute the number of active constraints per constraint type
        activeConstraints(i) = constraints_[i]->get_num_active_constraints();
    }
    int numActiveConstraints = activeConstraints.sum(); // Total number of active constraints
    // Collect triplets from all constraints, adjusting row indices
    std::vector<Triplet<double>> tripletListA;
    tripletListA.reserve(numActiveConstraints * q.size()); // Upper estimate, might be too large
    int rowOffset = 0; // Offset for the row index in the triplet list
    for (std::size_t i = 0; i < constraints_.size(); i++) {
        // Active constraints of the current constraint type
        int numActive = activeConstraints(i);
        if (numActive == 0) continue; // Skip if no active constraints
        // Get local triplet list from this constraint
        std::vector<Triplet<double>> tripletListLocal = constraints_[i]->compute_constraint_gradient(q, this->timeIntegrator_.get_qPrev(), dt); // Computes the constraint matrix dg/dx, each row is a constraint gradient for on DoF.
        for (const auto& triplet : tripletListLocal) {
            // Adjust row index to account for stacking
            tripletListA.emplace_back(rowOffset + triplet.row(), triplet.col(), triplet.value());
        }
        rowOffset += numActive; // Update offset for next constraint type
    }
    // Create the sparse matrix from the triplet list
    SparseMatrix<double> constraintMatrix(numActiveConstraints, q.size());
    constraintMatrix.setFromTriplets(tripletListA.begin(), tripletListA.end());
    return constraintMatrix;
}


template<int vertexDim, int elementDim>
bool Energy<vertexDim, elementDim>::update_active_constraints(VectorXd& q) const {
    // Loop through all constraints and update the active constraints 
    for (std::size_t i = 0; i < constraints_.size(); i++) {
        // Update the active constraints if fixedConstraintsFlag_ is false
        if (!constraints_[i]->get_fixed_constraint_flag()) {
            bool updated = constraints_[i]->update_active_constraints(q);
            if (updated == false) {
                return false; // Error in updating active constraints
            }
        }
    }
    return true;
}


template<int vertexDim, int elementDim>
std::vector<VectorXd> Energy<vertexDim, elementDim>::compute_elementwise_energy (VectorXd& q, double dt, Params& actuation, const std::vector<std::string>& VisualizationOption) const {
    UNUSED(dt);
    
    // Compute the energy of the system based on the current vertex positions q.
    // Reservation for possible values
    unsigned int size = 1;
    if (VisualizationOption.size() == 0)
        std::cerr << "No energy type for visualization is specified. \n";
    else
        size = VisualizationOption.size();

    // Compute energy contributions for each vertex individually due to external vertex forces
    MatrixXd externalForceMatrix;
    if (externalForcesFlag_) {
        // Compute total external forces TODO: Very inefficient in the local element loop!
        VectorXd externalForces = VectorXd::Zero(numVertices_ * vertexDim);
        for (std::size_t i = 0; i < this->externalForces_.size(); i++) {
            VectorXd act;
            if (this->externalForces_[i]->actuationFlag_) {
                act = actuation.get_value(this->externalForces_[i]->get_force_name()).transpose().reshaped();
            }
            else {
                act = VectorXd::Zero(0); // Default actuation if not specified
            }
            VectorXd externalForce = this->externalForces_[i]->compute_force(q, act);
            externalForces += externalForce;
        }
        externalForceMatrix = externalForces.reshaped<RowMajor>(numVertices_, vertexDim);
    }
    std::vector<VectorXd> elementEnergies(size, VectorXd::Zero(numElements_));
    #pragma omp parallel for
    for (int i = 0; i < numElements_; i++) {
        // Get the element vertex indices
        VectorXi ele = eleIdx_.row(i);
        // Disassemble into the element vertices
        Matrix<double, elementDim, vertexDim> vertices;
        for (int j = 0; j < elementDim; j++) {
            for (int k = 0; k < vertexDim; k++) {
                vertices(j, k) = q(vertexDim * ele(j) + k);
            }
        }
        for (size_t VisualCount = 0; VisualCount < VisualizationOption.size(); VisualCount++) {           
            const std::string& name = VisualizationOption[VisualCount];
            // Compute the local element energy
            if(name == "all"){
                elementEnergies[VisualCount](i) = this->elements_[i]->compute_energy(vertices, actuation);
                if (externalForcesFlag_) {
                    for (int j = 0; j < elementDim; j++) {
                        elementEnergies[VisualCount](i) += externalForceMatrix.row(ele(j)).dot(q.template segment<3>(vertexDim * ele(j))-this->timeIntegrator_.get_qPrev().template segment<3>(vertexDim * ele(j)));
                    }
                }
            } else {
                elementEnergies[VisualCount](i) = this->elements_[i]->compute_specific_energy(vertices, actuation, name);
            }
        }
    }
    return elementEnergies;
}


template<int vertexDim, int elementDim>
void Energy<vertexDim, elementDim>::update_state (VectorXd& q, double dt) {
    // Update the acceleration, velocity and position.
    this->timeIntegrator_.update_state(q, dt);
    if(localEnergyUpdateFlag_){
        for (int i = 0; i < numElements_; i++) {
            // Get the element vertex indices
            VectorXi ele = eleIdx_.row(i);
            // Disassemble into the element vertices
            Matrix<double, elementDim, vertexDim> vertices;
            Matrix<double, elementDim, vertexDim> acceleration;
            for (int j = 0; j < elementDim; j++) {
                for (int k = 0; k < vertexDim; k++) {
                    vertices(j, k) = q(vertexDim * ele(j) + k);
                }
            }
            this->elements_[i]->update_state(vertices, dt);
        }
    }
}


template<int vertexDim, int elementDim>
void Energy<vertexDim, elementDim>::verbose_print_physics_properties() const {
    std::cout << "\n============================== Physics Properties ==============================\n";
    std::cout << std::left << std::setw(25) << "Num Elements:" << numElements_ << "\n";
    std::cout << std::left << std::setw(25) << "Num Vertices:" << numVertices_ << "\n";
    // Print unique element energy types
    std::set<std::string> uniqueEnergyTypes;
    for (const auto& energySet : elementEnergiesList_) {
        uniqueEnergyTypes.insert(energySet.begin(), energySet.end());
    }
    if (uniqueEnergyTypes.empty()) {
        std::cout << std::setw(25) << "Element Energies:" << "none\n";
    } else {
        auto it = uniqueEnergyTypes.begin();
        std::cout << std::setw(25) << "Element Energies:" << *it << "\n";
        for (++it; it != uniqueEnergyTypes.end(); ++it) {
            std::cout << std::setw(25) << "" << *it << "\n";
        }
    }
    // Print constraints
    if (constraintTypesList_.empty()) {
        std::cout << std::setw(25) << "Constraints:" << "none\n";
    } else {
        auto it = constraintTypesList_.begin();
        std::cout << std::setw(25) << "Constraints:" << *it << "\n";
        for (++it; it != constraintTypesList_.end(); ++it) {
            std::cout << std::setw(25) << "" << *it << "\n";
        }
    }
    // External force flag
    std::cout << std::setw(25) << "External Forces:" 
              << (externalForcesFlag_ ? "applied" : "not applied") << "\n";

    std::cout << "================================================================================\n";
}

template class Energy<3, 8>;
template class Energy<3, 4>;
