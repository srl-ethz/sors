#include "hexahedron.h"

#define NQUADPOINTS_HEX 8

Hexahedron::Hexahedron(
    Params parameterMap,
    const Matrix<double, HEX_E_DIM, HEX_V_DIM>& undeformedVertices,
    const Vector<double, HEX_V_DIM>& gravAcceleration,
    std::set<std::string> elementEnergiesStringSet
) : Element<HEX_V_DIM, HEX_E_DIM>(undeformedVertices, parameterMap.get_value("density")(0,0), compute_volume(undeformedVertices), NQUADPOINTS_HEX),
    shapeFunctionsGradients_(compute_shape_functions_gradients(undeformedVertices)) {
    // Initialize deformation hessians using shapeFunctionsGradients_
    this->deformationHessians_ = compute_deformation_hessians();

    // Create 9 3x3 unit matrices for Hessian computation.
    std::array<Matrix<double, HEX_V_DIM, HEX_V_DIM>, 9> unitMatrices;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Matrix<double, HEX_V_DIM, HEX_V_DIM> unitMatrix = Matrix<double, HEX_V_DIM, HEX_V_DIM>::Zero();
            unitMatrix(i, j) = 1;
            unitMatrices[3*i + j] = unitMatrix;
        }
    }

    // Creating energy objects for different energy types. We now convert the set of strings to a vector of unique pointers to elementEnergy objects.
    elementEnergies_ = std::vector<std::unique_ptr<ElementEnergy<HEX_V_DIM, HEX_E_DIM>>>(elementEnergiesStringSet.size());
    elementEnergiesString_ = std::vector<std::string>(elementEnergiesStringSet.size());
    int energyIdx = 0;
    for (std::set<std::string>::iterator it = elementEnergiesStringSet.begin(); it != elementEnergiesStringSet.end(); ++it) {
        if (*it == "gravitational") {
            double density = parameterMap.get_value("density")(0,0);
            elementEnergies_[energyIdx] = std::make_unique<GravitationalEnergy<HEX_V_DIM, HEX_E_DIM>>(gravAcceleration, density, unitMatrices);
            elementEnergiesString_[energyIdx] = *it;
        }
        else if (*it == "neohookean") {
            // Create an elementEnergy instance for neohookean elastic energy
            double youngsModulus = parameterMap.get_value("youngsModulus")(0,0);
            double poissonsRatio = parameterMap.get_value("poissonsRatio")(0,0);
            elementEnergies_[energyIdx] = std::make_unique<NeoHookeanEnergy<HEX_V_DIM, HEX_E_DIM>>(youngsModulus, poissonsRatio, unitMatrices); 
            elementEnergiesString_[energyIdx] = *it;
        }
        else if (*it == "stableneohookean") {
            // Create an elementEnergy instance for stable neohookean elastic energy
            double youngsModulus = parameterMap.get_value("youngsModulus")(0,0);
            double poissonsRatio = parameterMap.get_value("poissonsRatio")(0,0);
            elementEnergies_[energyIdx] = std::make_unique<StableNeoHookeanEnergy<HEX_V_DIM,HEX_E_DIM>>(youngsModulus, poissonsRatio, unitMatrices); 
            elementEnergiesString_[energyIdx] = *it;
        }
        else if (*it == "pseudostrain") {
            // Create an elementEnergy instance for muscle elastic energy
            double b = parameterMap.get_value("b")(0,0);
            double c = parameterMap.get_value("c")(0,0);
            double D = parameterMap.get_value("D")(0,0);
            elementEnergies_[energyIdx] = std::make_unique<PseudostrainEnergy<HEX_V_DIM,HEX_E_DIM>>(b, c,D, unitMatrices); 
            elementEnergiesString_[energyIdx] = *it;
        }
        else if (*it == "softconMuscle") {
            // Create an elementEnergy instance for softcon muscle energy
            int muscleGroup = parameterMap.get_value("muscleGroup")(0,0);
            double muscleStiffness = parameterMap.get_value("softconStiffness")(0,0);
            VectorXd muscleDirection = parameterMap.get_value("softconDirection").reshaped<RowMajor>();
            elementEnergies_[energyIdx] = std::make_unique<SoftconMuscleEnergy<HEX_V_DIM, HEX_E_DIM>>(muscleGroup, muscleStiffness, muscleDirection, unitMatrices, deformationHessians_);
            elementEnergiesString_[energyIdx] = *it;
        }
        else {
            std::cout << bcolors.FAIL << "Unknown energy type: " << *it << bcolors.ENDC << std::endl;
            exit(1);
        }
        energyIdx++;
    }
}


Vector<double, HEX_EV_DIM> Hexahedron::compute_gradient(
    Matrix<double, HEX_E_DIM, HEX_V_DIM>& vertices,
    Params& actuation) const {

    // Compute tetrahedron specific properties
    const std::vector<Eigen::Matrix<double, HEX_V_DIM, HEX_V_DIM>> deformationGradients = compute_deformation_gradients(vertices);
    Vector<double, HEX_EV_DIM> totalGradient = Vector<double, HEX_EV_DIM>::Zero();
    for (std::size_t i = 0; i < elementEnergies_.size(); i++) {
        // Compute energy gradient
        // Find actuation where needed
        MatrixXd act; 
        if (this->elementEnergies_[i]->actuationFlag_) act = actuation.get_value(this->elementEnergies_[i]->get_energy_name()); 
        else act = MatrixXd::Zero(0, 0); 
        totalGradient += elementEnergies_[i]->compute_gradient(vertices, deformationGradients, deformationHessians_, act) * undeformedVolume_;
    }
    return totalGradient;
}


Matrix<double, HEX_EV_DIM, HEX_EV_DIM> Hexahedron::compute_hessian(
    Matrix<double, HEX_E_DIM, HEX_V_DIM>& vertices,
    Params& actuation) const {

    // Compute hexahedron specific properties
    const std::vector<Eigen::Matrix<double, HEX_V_DIM, HEX_V_DIM>> deformationGradients = compute_deformation_gradients(vertices);
        Matrix<double, HEX_EV_DIM, HEX_EV_DIM> totalHessian = Matrix<double, HEX_EV_DIM, HEX_EV_DIM>::Zero();
        for (std::size_t i = 0; i < elementEnergies_.size(); i++) {
            // Compute energy hessians
            // Find actuation where needed
            MatrixXd act; 
            if (this->elementEnergies_[i]->actuationFlag_) act = actuation.get_value(this->elementEnergies_[i]->get_energy_name()); 
            else act = MatrixXd::Zero(0, 0); 
            totalHessian += elementEnergies_[i]->compute_hessian(vertices, deformationGradients, deformationHessians_, act) * undeformedVolume_;
        }
    return totalHessian;
}


double Hexahedron::compute_volume(const Matrix<double, HEX_E_DIM, HEX_V_DIM>& vertices) const {
    // Convert the hexahedron in 5 tetrahedron and compute the mass of each of them
    // Important: the order of the vertex must be the following: bottom_face: (0, 1, 2, 3); top_face (4, 5, 6, 7) --> connected as 0-4; 1-5; 2-6; 3-7
    Matrix<int, 5, 4> tets_local_idxs;
    tets_local_idxs.row(0) << 0, 1, 3, 4;
    tets_local_idxs.row(1) << 1, 2, 3, 6;
    tets_local_idxs.row(2) << 1, 4, 5, 6;
    tets_local_idxs.row(3) << 3, 4, 6, 7;
    tets_local_idxs.row(4) << 1, 3, 4, 6;
    
    // Initialize the volume at zero
    double volume = 0;
    for (int i = 0; i < tets_local_idxs.rows(); ++i) {
        // Volume of a tetrahedron: V = 1/6 * |(b - a) . ((c - a) x (d - a))|
        // a, b, c, d are the vertices of the tetrahedron
        Vector<double, HEX_V_DIM> ab = (vertices.row(tets_local_idxs(i, 1)) - vertices.row(tets_local_idxs(i, 0))).transpose();
        Vector<double, HEX_V_DIM> ac = (vertices.row(tets_local_idxs(i, 2)) - vertices.row(tets_local_idxs(i, 0))).transpose();
        Vector<double, HEX_V_DIM> ad = (vertices.row(tets_local_idxs(i, 3)) - vertices.row(tets_local_idxs(i, 0))).transpose();

        double tetVolume = std::abs(ab.dot(ac.cross(ad))) / 6.0;
        assert(((void)"Volume of element must be positive!", tetVolume > 0.0));
        volume += tetVolume;
    }
    return volume;
}


std::vector<Matrix<double, HEX_V_DIM, HEX_V_DIM>> Hexahedron::compute_deformation_gradients(Matrix<double, HEX_E_DIM, HEX_V_DIM>& vertices) const {
    /* Fs = X_nod * G_s for s = 0, ...,7 --> we have a deformation gradient for each quadrature point
     *  X_nod: deformed vertices 3x8
     *  G_s = shape_functions_gradients 8x3 */
    std::vector<Matrix<double, HEX_V_DIM, HEX_V_DIM>> deformationGradients(nquadpoints_);
    for (int s = 0; s < nquadpoints_; ++s) {
        deformationGradients[s] = vertices.transpose() * shapeFunctionsGradients_[s];
    }
    return deformationGradients; 
}
 

std::vector<Matrix<double, HEX_V_DIM*HEX_V_DIM, HEX_V_DIM*HEX_E_DIM>> Hexahedron::compute_deformation_hessians() {
    // Calculate deformation hessians for each quadrature point: 8 9x24 matrixes
    // First define 24 unitary matrices 3x8
    std::vector<Matrix<double, HEX_V_DIM, HEX_E_DIM>> blockMatrices;
    for (int i = 0; i < HEX_E_DIM; ++i) {
        for (int j = 0; j < HEX_V_DIM; ++j) {
            Matrix<double, HEX_V_DIM, HEX_E_DIM> unitMatrix = Matrix<double, HEX_V_DIM, HEX_E_DIM>::Zero();
            unitMatrix(j, i) = 1.;
            blockMatrices.push_back(unitMatrix);
        }
    }
    std::vector<Matrix<double, HEX_V_DIM*HEX_V_DIM, HEX_V_DIM*HEX_E_DIM>> deformationHessians(nquadpoints_);
    // Compute one deformation hessian for each quadrature point
    for (int s = 0; s < nquadpoints_; ++s) {
        for (std::size_t i = 0; i < blockMatrices.size(); ++i) {
            // Multiply each matrix with gradients of shape functions 3x8 times 8x3
            Matrix<double, HEX_V_DIM, HEX_V_DIM> dFdx_i = blockMatrices[i] * shapeFunctionsGradients_[s];
            // Flatten each matrix row-wise, put it as a column in the final 9x12 matrix (deformation Hessian dF/dx)
            deformationHessians[s].col(i) = dFdx_i.reshaped<RowMajor>();
        }
    }
    return deformationHessians; 
}


std::vector<Matrix<double, HEX_E_DIM, HEX_V_DIM>> Hexahedron::compute_shape_functions_gradients(const Matrix<double, HEX_E_DIM, HEX_V_DIM>& undeformedVert) {
    Eigen::Matrix<double, 3, 8> unitaryCubeNodes; 
    unitaryCubeNodes.col(0) << 1., -1., -1.;
    unitaryCubeNodes.col(1) << 1., 1., -1.;
    unitaryCubeNodes.col(2) << -1., 1., -1.;
    unitaryCubeNodes.col(3) << -1., -1., -1.;
    unitaryCubeNodes.col(4) << 1., -1., 1.;
    unitaryCubeNodes.col(5) << 1., 1., 1.;
    unitaryCubeNodes.col(6) << -1., 1., 1.;
    unitaryCubeNodes.col(7) << -1., -1., 1.;
    Eigen::Matrix<double, 3, 8> unitaryCubeQuadratureNodes = (1. / std::sqrt(3.)) * unitaryCubeNodes; // Gauss quadrature points --> more precise than cube nodes
    // Eigen::Matrix<double, 3, 8> unitaryCubeQuadratureNodes = unitaryCubeNodes; //in case you want to test cube nodes as quadrature points

    // Gradient of the shape functions on the unitary cube
    std::vector<Matrix<double, 8, 3>> hatG(nquadpoints_); 

    // Compute the gradient of the REFERENCE shape functions on the unitary cube
    // NB: s is the index for the quadrature node, i is the index for the vertices of the unitary cube and the index of the shape function:
    // N_hat_i(csi_x, csi_y, csi_z) = (1 + (csi_x)_i * csi_x) * (1 + (csi_y)_i * csi_y) * (1 + (csi_z)_i * csi_z).
    // We need to compute the gradient w.r.t. csi_x, csi_y, csi_z and then evaluate the three expressions on the the quadrature points.
    for (int s = 0; s < nquadpoints_; ++s) {
        const double csiQuadNodesCoordx = unitaryCubeQuadratureNodes(0, s), csiQuadNodesCoordy = unitaryCubeQuadratureNodes(1, s), csiQuadNodesCoordz = unitaryCubeQuadratureNodes(2, s);
        for (int i = 0; i < 8; ++i) {
            const double csiNodeiCoordx = unitaryCubeNodes(0, i), csiNodeiCoordy = unitaryCubeNodes(1, i), csiNodeiCoordz = unitaryCubeNodes(2, i);
            hatG[s].row(i) = 0.125 * Vector3d(csiNodeiCoordx * (1 + csiNodeiCoordy*csiQuadNodesCoordy) * (1 + csiNodeiCoordz*csiQuadNodesCoordz),
                                               csiNodeiCoordy * (1 + csiNodeiCoordx*csiQuadNodesCoordx) * (1 + csiNodeiCoordz*csiQuadNodesCoordz),
                                               csiNodeiCoordz * (1 + csiNodeiCoordx*csiQuadNodesCoordx) * (1 + csiNodeiCoordy*csiQuadNodesCoordy)); // This formula is hex-specific and can be found in the literature
        }
    }
    std::vector<Matrix<double, HEX_E_DIM, HEX_V_DIM>> shapeFunctionsGradients(nquadpoints_);
    // Compute the actual gradients of the shape functions
    for (int s = 0; s < nquadpoints_; ++s) {
        // First compute the deformation gradient (hatF) of the map: csi -> X
        Eigen::Matrix<double, HEX_V_DIM, HEX_V_DIM> hatF = undeformedVert.transpose() * hatG[s];
        shapeFunctionsGradients[s] = (hatF.inverse().transpose() * hatG[s].transpose()).transpose();
    }
    return shapeFunctionsGradients;
}


Vector<double, HEX_V_DIM> Hexahedron::compute_com(const Matrix<double, HEX_E_DIM, HEX_V_DIM>& vertices) const {
    // Convert the hexahedron in 5 tetrahedron and compute weighted center of mass
    // Important: the order of the vertex must be the following: bottom_face: (0, 1, 2, 3); top_face (4, 5, 6, 7) --> connected as 0-4; 1-5; 2-6; 3-7
    Matrix<int, 5, 4> tetsLocalIdxs;
    tetsLocalIdxs.row(0) << 0, 1, 3, 4;
    tetsLocalIdxs.row(1) << 1, 2, 3, 6;
    tetsLocalIdxs.row(2) << 1, 4, 5, 6;
    tetsLocalIdxs.row(3) << 3, 4, 6, 7;
    tetsLocalIdxs.row(4) << 1, 3, 4, 6;
    
    // Initialize the center of mass of the hexahedron
    Vector<double, HEX_V_DIM> centerOfMass = Vector<double, HEX_V_DIM>::Zero();
    double volume = 0;

    for (int i = 0; i < tetsLocalIdxs.rows(); ++i) {
        // Volume of a tetrahedron: V = 1/6 * |(b - a) . ((c - a) x (d - a))|
        // a, b, c, d are the vertices of the tetrahedron
        const Vector<double, HEX_V_DIM> a = vertices.row(tetsLocalIdxs(i, 0)).transpose();
        const Vector<double, HEX_V_DIM> b = vertices.row(tetsLocalIdxs(i, 1)).transpose();
        const Vector<double, HEX_V_DIM> c = vertices.row(tetsLocalIdxs(i, 2)).transpose();
        const Vector<double, HEX_V_DIM> d = vertices.row(tetsLocalIdxs(i, 3)).transpose();

        const auto ab = b - a;
        const auto ac = c - a;
        const auto ad = d - a;

        double tetVolume = std::abs(ab.dot(ac.cross(ad))) / 6.0;
        assert(((void)"Volume of element must be positive!", tetVolume > 0.0));
        volume += tetVolume;

        const Vector<double, HEX_V_DIM> tet_centerOfMass = (a + b + c + d) / 4.0;
        centerOfMass += tetVolume * tet_centerOfMass;
    }
    return centerOfMass / volume;
}
