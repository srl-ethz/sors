#include "tetrahedron.h"

#define NQUADPOINTS_TET 1

Tetrahedron::Tetrahedron(
    Params parameterMap,
    const Matrix<double, TET_E_DIM, TET_V_DIM>& undeformedVertices,
    std::set<std::string> elementEnergiesStringSet
) : Element<TET_V_DIM, TET_E_DIM>(undeformedVertices, parameterMap.get_value("density")(0,0), compute_volume(undeformedVertices), NQUADPOINTS_TET) {
    // Calculate inverted reference shape matrix for deformation gradients (3x3 matrix)
    Matrix<double, TET_V_DIM, TET_V_DIM> refShapeMatrix;
    refShapeMatrix.col(0) = undeformedVertices.row(0) - undeformedVertices.row(3);
    refShapeMatrix.col(1) = undeformedVertices.row(1) - undeformedVertices.row(3);
    refShapeMatrix.col(2) = undeformedVertices.row(2) - undeformedVertices.row(3);
    invRefShapeMatrix_ = refShapeMatrix.inverse();

    // Calculate deformation hessian (constant, only depends on undeformed vertices, 9x12 matrix)
    // Contains 12 3x3 matrices, the first nine are unit matrices, the last three have -1 on each row.
    std::vector<Matrix<double, TET_V_DIM, TET_V_DIM>> blockMatrices;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Matrix<double, TET_V_DIM, TET_V_DIM> unitMatrix = Matrix<double, TET_V_DIM, TET_V_DIM>::Zero();
            unitMatrix(j, i) = 1;
            blockMatrices.push_back(unitMatrix);
        }
    }
    blockMatrices.push_back((Matrix<double, TET_V_DIM, TET_V_DIM>() << -1, -1, -1, 0, 0, 0, 0, 0, 0).finished());
    blockMatrices.push_back((Matrix<double, TET_V_DIM, TET_V_DIM>() << 0, 0, 0, -1, -1, -1, 0, 0, 0).finished());
    blockMatrices.push_back((Matrix<double, TET_V_DIM, TET_V_DIM>() << 0, 0, 0, 0, 0, 0, -1, -1, -1).finished());

    Matrix<double, TET_V_DIM * TET_V_DIM, TET_E_DIM * TET_V_DIM> deformationHessian;
    for (std::size_t i = 0; i < blockMatrices.size(); ++i) {
        // Multiply each matrix with inverted reference shape matrix
        blockMatrices[i] *= invRefShapeMatrix_;
        // Flatten each matrix row-wise, put it as a column in the final 9x12 matrix (deformation hessian dF/dx)
        deformationHessian.col(i) = blockMatrices[i].reshaped<RowMajor>();
    }

    // Store the deformation hessian in the array with only one element as we are using only one quadrature point
    deformationHessians_.resize(nquadpoints_);
    for (int s = 0; s < nquadpoints_; ++s) {
        deformationHessians_[s] = deformationHessian;
    }

    // Create 9 3x3 unit matrices for Hessian computation.
    std::array<Matrix<double, TET_V_DIM, TET_V_DIM>, 9> unitMatrices;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Matrix<double, TET_V_DIM, TET_V_DIM> unitMatrix = Matrix<double, TET_V_DIM, TET_V_DIM>::Zero();
            unitMatrix(i, j) = 1;
            unitMatrices[3*i + j] = unitMatrix;
        }
    }

    // Creating energy objects for different energy types. We now convert the set of strings to a vector of unique pointers to elementEnergy objects.
    elementEnergies_ = std::vector<std::unique_ptr<ElementEnergy<TET_V_DIM, TET_E_DIM>>>(elementEnergiesStringSet.size());
    elementEnergiesString_ = std::vector<std::string>(elementEnergiesStringSet.size());
    int energyIdx = 0;
    for (std::set<std::string>::iterator it = elementEnergiesStringSet.begin(); it != elementEnergiesStringSet.end(); ++it) {
        if (*it == "neohookean") {
            // Create an elementEnergy instance for neohookean elastic energy
            double youngsModulus = parameterMap.get_value("youngsModulus")(0,0);
            double poissonsRatio = parameterMap.get_value("poissonsRatio")(0,0);
            elementEnergies_[energyIdx] = std::make_unique<NeoHookeanEnergy<TET_V_DIM, TET_E_DIM>>(youngsModulus, poissonsRatio, unitMatrices); 
            elementEnergiesString_[energyIdx] = *it;
        }
        else if (*it == "stableneohookean") {
            // Create an elementEnergy instance for stable neohookean elastic energy
            double youngsModulus = parameterMap.get_value("youngsModulus")(0,0);
            double poissonsRatio = parameterMap.get_value("poissonsRatio")(0,0);
            elementEnergies_[energyIdx] = std::make_unique<StableNeoHookeanEnergy<TET_V_DIM,TET_E_DIM>>(youngsModulus, poissonsRatio, unitMatrices); 
            elementEnergiesString_[energyIdx] = *it;
        }
        else if (*it == "pseudostrain") {
            // Create an elementEnergy instance for muscle elastic energy
            double b = parameterMap.get_value("b")(0,0);
            double c = parameterMap.get_value("c")(0,0);
            double D = parameterMap.get_value("D")(0,0);
            elementEnergies_[energyIdx] = std::make_unique<PseudostrainEnergy<TET_V_DIM,TET_E_DIM>>(b, c,D, unitMatrices); 
            elementEnergiesString_[energyIdx] = *it;
        }
        else if (*it == "softconMuscle") {
            // Create an elementEnergy instance for softcon muscle energy
            int muscleGroup = parameterMap.get_value("muscleGroup")(0,0);
            double muscleStiffness = parameterMap.get_value("softconStiffness")(0,0);
            VectorXd muscleDirection = parameterMap.get_value("softconDirection").reshaped<RowMajor>();
            elementEnergies_[energyIdx] = std::make_unique<SoftconMuscleEnergy<TET_V_DIM, TET_E_DIM>>(muscleGroup, muscleStiffness, muscleDirection, unitMatrices);
            elementEnergiesString_[energyIdx] = *it;
        }
        else {
            std::cout << bcolors.FAIL << "Unknown energy type: " << *it << bcolors.ENDC << std::endl;
            exit(1);
        }
        energyIdx++;
    }
}


Vector<double, TET_EV_DIM> Tetrahedron::compute_gradient(
    Matrix<double, TET_E_DIM, TET_V_DIM>& vertices,
    Params& actuation) const {
    // Compute tetrahedron specific properties
    const std::vector<Eigen::Matrix<double, TET_V_DIM, TET_V_DIM>> deformationGradients = compute_deformation_gradients(vertices);

    Vector<double, TET_EV_DIM> totalGradient = Vector<double, TET_EV_DIM>::Zero();
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


Matrix<double, TET_EV_DIM, TET_EV_DIM> Tetrahedron::compute_hessian(
    Matrix<double, TET_E_DIM, TET_V_DIM>& vertices,
    Params& actuation) const {
    // Compute tetrahedron specific properties
    const std::vector<Eigen::Matrix<double, TET_V_DIM, TET_V_DIM>> deformationGradients = compute_deformation_gradients(vertices);

        Matrix<double, TET_EV_DIM, TET_EV_DIM> totalHessian = Matrix<double, TET_EV_DIM, TET_EV_DIM>::Zero();
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


double Tetrahedron::compute_volume(const Matrix<double, TET_E_DIM, TET_V_DIM>& vertices) const {
    // Volume of a tetrahedron: V = 1/6 * |(b - a) . ((c - a) x (d - a))|
    // a, b, c, d are the vertices of the tetrahedron
    Vector<double, TET_V_DIM> ab = vertices.row(1) - vertices.row(0);
    Vector<double, TET_V_DIM> ac = vertices.row(2) - vertices.row(0);
    Vector<double, TET_V_DIM> ad = vertices.row(3) - vertices.row(0);

    double volume = std::abs(ab.dot(ac.cross(ad))) / 6.0;

    assert(((void)"Volume of element must be positive!", volume > 0.0));

    return volume;
}


std::vector<Matrix<double, TET_V_DIM, TET_V_DIM>> Tetrahedron::compute_deformation_gradients(Matrix<double, TET_E_DIM, TET_V_DIM>& vertices) const {
    // F = D_s * inv(D_m)
    // D_s: deformed shape matrix, D_m: reference shape matrix
    // See derivation in: SIGGRAPH 2012 Course FEM Simulation of 3D Deformable Solids: A practitioner’s guide to theory, discretization and model reduction.
    // Eftychios D. Sifakis University of Wisconsin-Madison, Version 1.0 [10 July 2012]
    Matrix<double, TET_V_DIM, TET_V_DIM> defShapeMatrix;
    defShapeMatrix.col(0) = vertices.row(0) - vertices.row(3);
    defShapeMatrix.col(1) = vertices.row(1) - vertices.row(3);
    defShapeMatrix.col(2) = vertices.row(2) - vertices.row(3);

    Matrix<double, TET_V_DIM, TET_V_DIM> deformationGradient = defShapeMatrix * invRefShapeMatrix_;

    std::vector<Matrix<double, TET_V_DIM, TET_V_DIM>> deformationGradients(nquadpoints_);
    for (int s = 0; s < nquadpoints_; ++s) {
        deformationGradients[s] = deformationGradient;
    }
    return deformationGradients; 
}


Vector<double, TET_V_DIM> Tetrahedron::compute_com(const Matrix<double, TET_E_DIM, TET_V_DIM>& vertices) const {
    Vector<double, TET_V_DIM> centerOfMass = vertices.colwise().sum() / vertices.rows();
    return centerOfMass;
}
