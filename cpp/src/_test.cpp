#include "_test.h"

int TestClass::testCount = 0;
int TestClass::testPassed = 0;
int TestClass::testFailed = 0;

// ---------- TESTING TESTAHEDRON ---------- //
Testahedron::Testahedron (Matrix<double, TET_E_DIM, TET_V_DIM>& vertices) : Tetrahedron(
    Params({"density", "youngsModulus", "poissonsRatio"}, {1.0, 1.0, 0.3}), 
vertices, std::set<std::string>{"neohookean"}) {}

double Testahedron::compute_energy (
    Matrix<double, TET_E_DIM, TET_V_DIM>& vertices,
    Params& actuation) const {

    UNUSED(actuation);
    
    // Compute the energy of the tetrahedron
    double energy = 0.0;

    // Dummy function x^2 * (x^2 - 0.5)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            energy += vertices(i, j) * vertices(i, j) * (vertices(i, j) * vertices(i, j) - 0.5);
        }
    }
    return energy;
}


Vector<double, TET_EV_DIM> Testahedron::compute_gradient (
    Matrix<double, TET_E_DIM, TET_V_DIM>& vertices,
    Params& actuation) const {

    UNUSED(actuation);

    // Compute the gradient of the energy of the tetrahedron
    Vector<double, 12> gradient = Vector<double, 12>::Zero();

    // Dummy function gradient (2x-1)*(2x+1)*x
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            gradient(i * 3 + j) = (2 * vertices(i, j) - 1) * (2 * vertices(i, j) + 1) * vertices(i, j);
        }
    }
    return gradient;
}


Matrix<double, TET_EV_DIM, TET_EV_DIM> Testahedron::compute_hessian (
    Matrix<double, TET_E_DIM, TET_V_DIM>& vertices,
    Params& actuation) const {

    UNUSED(actuation);

    // Compute the hessian of the energy of the tetrahedron
    Matrix<double, 12, 12> hessian = Matrix<double, 12, 12>::Zero();

    // Dummy function hessian 12x^2 - 1
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            hessian(i * 3 + j, i * 3 + j) = 12 * vertices(i, j) * vertices(i, j) - 1;
        }
    }
    return hessian;
}


void TestClass::add_result(bool result, std::string testName) {
    // Add the result of the test to the test count
    testCount++;
    if (result) {
        testPassed++;
        std::cout << bcolors.OKCYAN << "TEST [" << testCount << "/" << MAX_TESTS << "] PASSED: " << testName << bcolors.ENDC << std::endl;
    } else {
        testFailed++;
        std::cout << bcolors.FAIL << "TEST [" << testCount << "/" << MAX_TESTS << "] FAILED: " << testName << bcolors.ENDC << std::endl;
    }
}


// ---------- TESTING ENERGY ---------- //
TestEnergy::TestEnergy() : vertices_(5, 3), eleIdx_(2, 4) {
    // Initialize the mesh
    vertices_ << 0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        0, -1, 0;

    eleIdx_ << 0, 1, 2, 3,
        0, 1, 4, 3;

    std::vector<std::set<std::string>> elementEnergiesList(eleIdx_.rows());
    std::vector<Params> elementParameterList(eleIdx_.rows());
    for (auto& parameterMap : elementParameterList) {
        parameterMap.set_param("density", 1.0);
    }
    SimulationSettings settings;
    systemEnergy_ = Energy<VDIM, EDIM>(vertices_, eleIdx_, settings, {0.0, 0.0, 0.0}, 0.0, elementEnergiesList, elementParameterList);

    // Dummy vertices fixed for all Testahedrons
    Matrix<double, TET_E_DIM, TET_V_DIM> dummyVertices;
    dummyVertices << 0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1;

    // Now we overwrite the Tetrahedral elements with our test elements for dummy testing of energy computation
    systemEnergy_.elements_[0] = std::make_unique<Testahedron>(dummyVertices);
    systemEnergy_.elements_[1] = std::make_unique<Testahedron>(dummyVertices);
}


bool TestEnergy::test_init() {
    // Check that initialization works by seeing if the vertices of the last element match
    std::vector<std::set<std::string>> elementEnergiesList(eleIdx_.rows());
    std::vector<Params> elementParameterList(eleIdx_.rows());
    for (auto& parameterMap : elementParameterList) {
        parameterMap.set_param("density", 1.0);
    }

    SimulationSettings settings;
    Energy<VDIM, EDIM> systemEnergy(vertices_, eleIdx_, settings, {0.0, 0.0, 0.0}, 0.0, elementEnergiesList, elementParameterList);

    Matrix<double, -1, 3> vertGT(4, 3);
    vertGT << 0, 0, 0,
        1, 0, 0,
        0, -1, 0,
        0, 0, 1;
    Matrix<double, TET_E_DIM, TET_V_DIM> vertTet = systemEnergy.elements_[1]->get_undeformed_vertices();
    return (vertTet.isApprox(vertGT));
}


bool TestEnergy::test_energy() {
    // Dummy energy has three overlapping/shared nodes
    // Dummy energy function x^2 * (x^2 - 0.5)
    VectorXd reshaped_vertices = vertices_.reshaped<RowMajor>();
    Params actuation;
    double energy = systemEnergy_.compute_energy(reshaped_vertices, 0.0, actuation);
    return (energy == 3);
}


bool TestEnergy::test_gradient() {
    // Gradient should be gradient (2x-1)*(2x+1)*x
    Matrix<double, -1, 3> gradGT(5, 3);
    gradGT << 0, 0, 0,
        6, 0, 0,
        0, 3, 0,
        0, 0, 6,
        0, -3, 0;
    VectorXd reshapedVertices = vertices_.reshaped<RowMajor>();
    Params actuation;
    Vector<double, NUM_VERTICES * VDIM> gradient = systemEnergy_.compute_gradient(reshapedVertices, 0.0, actuation);
    std::cout << "Grad: " << gradient.reshaped<RowMajor>(NUM_VERTICES, VDIM) << std::endl;
    return (gradient.isApprox(gradGT.reshaped<RowMajor>()));
}


bool TestEnergy::test_hessian() {
    // Hessian should be a diagonal matrix of hessian 12x^2 - 1
    Matrix<double, -1, 3> hessDiagGT(5, 3);
    hessDiagGT << -2, -2, -2,
        22, -2, -2,
        -1, 11, -1,
        -2, -2, 22,
        -1, 11, -1;
    Matrix<double, NUM_VERTICES * VDIM, NUM_VERTICES * VDIM> hess_GT = hessDiagGT.reshaped<RowMajor>().matrix().asDiagonal();
    VectorXd reshapedVertices = vertices_.reshaped<RowMajor>();
    Params actuation;
    Matrix<double, NUM_VERTICES * VDIM, NUM_VERTICES * VDIM> hessian = systemEnergy_.compute_hessian(reshapedVertices, 0.0, actuation);
    std::cout << "Hessian: " << hessian << std::endl;
    return (hessian.isApprox(hess_GT));
}


// ---------- TESTING OPTIMIZATION ---------- //
bool TestOptimization::test_optimization() {
    // Test the optimization class by reaching hand-crafted optimum.
    Matrix<double, -1, 3> sol_GT(5, 3);
    sol_GT << 0, 0, 0,
        0.5, 0, 0,
        0, 0.5, 0,
        0, 0, 0.5,
        0, -0.5, 0;
    Solver<VDIM, EDIM> solver;
    VectorXd reshapedVertices = vertices_.reshaped<RowMajor>();
    Params actuation;
    VectorXd solution = solver.minimize_newton(reshapedVertices, systemEnergy_, 1.0, actuation);
    // std::cout << "Solution: " << solution << std::endl;

    // NOTE: Multiple optima available, we check function evaluation
    // return (solution.isApprox(sol_GT.reshaped<RowMajor>()));

    // std::cout << "Final Gradient: " << system_energy_.computeGradient(solution).squaredNorm() << std::endl;
    return (systemEnergy_.compute_gradient(solution, 1.0, actuation).squaredNorm() < TOL);
}


// ---------- TESTING SINGLE TETRAHEDRON ---------- //
TestTetrahedron::TestTetrahedron() : dt_(1.0) {
    // Initialize our sample tetrahedron
    undeformedVertices_ << 0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1;

    currentVertices_ << 0, 0, 0,
        2, 0, 0,
        0, 1, 0,
        0, 0, 1;
}


bool TestTetrahedron::test_deformationHessian() {
    Params parameterMap;
    parameterMap.set_param("density", 1.0);
    parameterMap.set_param("youngsModulus", 1.0);
    parameterMap.set_param("poissonsRatio", 0.3);
    Tetrahedron tet(parameterMap, undeformedVertices_, std::set<std::string>{"neohookean"});
    Matrix<double, VDIM * VDIM, EDIM * VDIM> deformationHessian = tet.get_deformation_hessian();
    Matrix<double, VDIM * VDIM, EDIM * VDIM> deformationHessianCalc;
    deformationHessianCalc << -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    return deformationHessian.isApprox(deformationHessianCalc);
} 


bool TestTetrahedron::test_energy_zero() {
    Matrix<double, EDIM, VDIM> velocity;
    velocity << 0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0;
    Matrix<double, EDIM, VDIM> acceleration;
    acceleration << 0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0;

    Params parameterMap;
    parameterMap.set_param("density", 1.0);
    parameterMap.set_param("youngsModulus", 1.0);
    parameterMap.set_param("poissonsRatio", 0.3);
    Tetrahedron tet(parameterMap, undeformedVertices_, std::set<std::string>{"neohookean"});
    // Compute energy of the undeformed state.
    // Should be 0 since no movement, no deformation and no gravity.
    Params actuation;
    double energy = tet.compute_energy(undeformedVertices_, actuation);
    return (std::abs(energy) < TOL);
}


bool TestTetrahedron::test_gradient_zero() {
    Matrix<double, EDIM, VDIM> velocity;
    velocity << 0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0;
    Matrix<double, EDIM, VDIM> acceleration;
    acceleration << 0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0;

    Params parameterMap;
    parameterMap.set_param("density", 1.0);
    parameterMap.set_param("youngsModulus", 1.0);
    parameterMap.set_param("poissonsRatio", 0.3);
    Tetrahedron tet(parameterMap, undeformedVertices_, std::set<std::string>{"neohookean"});
    // Compute gradient of the undeformed state.
    // Should be 0 since no movement, no deformation and no gravity.
    Params actuation;
    Vector<double, EDIM * VDIM> gradient = tet.compute_gradient(undeformedVertices_, actuation);
    Vector<double, EDIM * VDIM> gradient_zero = Vector<double, EDIM * VDIM>::Zero();
    return (gradient.isApprox(gradient_zero));
}


// ---------- TESTING STANDARD CUBE ---------- //
TestCube::TestCube() : dt_(0.01) {
    // Initialize our cube from five tetrahedrons
    vertices_ = (MatrixXd(8, 3) << 0, 0, 0,
                 0, 1, 0,
                 1, 1, 0,
                 1, 0, 0,
                 0, 0, 1,
                 0, 1, 1,
                 1, 1, 1,
                 1, 0, 1)
                    .finished();

    elements_ = (MatrixXi(5, 4) << 0, 1, 3, 4,
                 1, 4, 5, 6,
                 1, 2, 3, 6,
                 3, 4, 6, 7,
                 1, 3, 4, 6)
                    .finished();
}


bool TestCube::test_energy_min() {
    // Apply pressure on top surface, elements 1 and 2
    std::vector<std::set<std::string>> elementEnergiesList(elements_.rows());   
    std::vector<Params> elementParameterList(elements_.rows());
    for (int i = 0; i < elements_.rows(); i++) {
        elementEnergiesList[i].insert("neohookean");

        // Parameter list
        elementParameterList[i].set_param("density", 1000.);
        elementParameterList[i].set_param("youngsModulus", 250000.0);
        elementParameterList[i].set_param("poissonsRatio", 0.4);
    }

    // Define constraint types (accepted types: "neumannBC", "planeContact", "maxDistance")
    // Set the required parameters (see constraint constructors) in constraintParameterList
    std::vector<std::string> constraintTypesList;
    Params constraintParameterList;
    // Constraint types
    constraintTypesList.push_back("neumannBC");
    // Constraint parameters
    std::vector<int> neumannBCmask(3*vertices_.rows(), 0);
    std::vector<double> neumannBCvalue(3*vertices_.rows(), 0);
    std::vector<int> fixedVertices = {0, 3, 4, 7};
    for (size_t i = 0; i < fixedVertices.size(); i++) {
        neumannBCmask[3*i] = 1; neumannBCvalue[3*i] = 0.0; 
        neumannBCmask[3*i+1] = 1; neumannBCvalue[3*i+1] = 0.0;
        neumannBCmask[3*i+2] = 1; neumannBCvalue[3*i+2] = 0.0;
    }
    
    // Set active constraints mask for each vertex in 3 dimensions
    constraintParameterList.set_param("neumannBCmask", neumannBCmask);
    // Set neumannBC value (velocity) for each vertex in 3 dimensions
    constraintParameterList.set_param("neumannBCvalue", neumannBCvalue);

    // Define force types (accepted types: "vertexForce", "pressure")
    std::vector<std::string> forceTypesList;
    Params forceParameterList;
    forceTypesList.push_back("pressure");
    forceParameterList.set_param("surfaceVertexIdx", (MatrixXd(2,3) << 1, 5, 6, 1, 6, 2).finished()); // vertex indices of the surface triangles
    forceParameterList.set_param("surfaceGroups", std::vector<double>{0, 0}); // group indices of the surface triangles

    // Define actuation signals (for energies, constraints and external forces)
    Params actuation;
    actuation.set_param("pressure", MatrixXd::Zero(1,1)); // Initialize with zeros
    
    SimulationSettings settings;
    settings.timeSteppingScheme = "backward_euler";
    settings.dt = dt_;
    settings.CFLtimeSteppingFlag = false;
    settings.solverMethod = "minimize_newton";
    Energy<3,4> systemEnergy = Energy<3, 4>(vertices_, elements_, settings, {0.0, 0.0, -9.81}, 0.0, elementEnergiesList, elementParameterList, constraintTypesList, constraintParameterList, forceTypesList, forceParameterList);
    Solver<3, 4> solver;

    VectorXd solution = vertices_.reshaped<RowMajor>();
    // IO::save_tet_VTU("output/test_0", vertices_, elements_);
    Matrix<double, -1, 3> reshapedSolution = solution.reshaped<RowMajor>(solution.size()/3, 3);

    const double maxPressure = 1000.0;
    const int numSteps = 10;
    for (int i = 1; i < numSteps; i++) {
        MatrixXd act = actuation.get_value("pressure");
        double pressure = 0.0;
        pressure = (maxPressure * 2*i/numSteps);
        if (pressure > maxPressure) {pressure = maxPressure;}
        act(0, 0) = pressure;
        actuation.set_param("pressure", act);

        // One step forward in time
        double startingEnergy = systemEnergy.compute_energy(solution, dt_, actuation);

        solution = solver.step(solution, systemEnergy, actuation);

        reshapedSolution = solution.reshaped<RowMajor>(solution.size() / 3, 3);
        double endingEnergy = systemEnergy.compute_energy(solution, dt_, actuation);
        // IO::save_tet_VTU("output/test_"+std::to_string(i), reshapedSolution, elements_);
        // std::cout << "Energy: " << startingEnergy << " -> " << endingEnergy << std::endl;
        if (endingEnergy > startingEnergy) {
            return false;
        }
    }
    return true;
}


bool TestCube::test_gravity_force() {
    std::vector<std::set<std::string>> elementEnergiesList(elements_.rows());   
    std::vector<Params> elementParameterList(elements_.rows());
    for (int i = 0; i < elements_.rows(); i++) {
        elementEnergiesList[i].insert("neohookean");

        // Parameter list
        elementParameterList[i].set_param("density", 1.);
        elementParameterList[i].set_param("youngsModulus", 500000.0);
        elementParameterList[i].set_param("poissonsRatio", 0.4);
    }

    SimulationSettings settings;
    settings.timeSteppingScheme = "backward_euler";
    settings.dt = dt_;
    settings.CFLtimeSteppingFlag = false;
    settings.solverMethod = "minimize_newton";

    Energy<3,4> systemEnergy = Energy<3, 4>(vertices_, elements_, settings, {0.0, 0.0, -10}, 0.0, elementEnergiesList, elementParameterList);
    Solver<3, 4> solver;

    VectorXd solution = vertices_.reshaped<RowMajor>();
    // IO::save_tet_VTU("output_test/test_0", vertices_, elements_);
    Matrix<double, -1, 3> reshapedSolution = solution.reshaped<RowMajor>(solution.size() / 3, 3);

    const int numSteps = 10;
    VectorXd ycom = VectorXd::Zero(numSteps);
    ycom(0) = reshapedSolution.colwise().mean()(2);
    

    for (int i = 1; i < numSteps; i++) {
        // One step forward in time
        solution = solver.step(solution, systemEnergy);

        reshapedSolution = solution.reshaped<RowMajor>(solution.size() / 3, 3);
        // IO::save_tet_VTU("output_test/test_"+std::to_string(i), reshapedSolution, elements_);

        // Tracking center of mass
        Vector3d centerOfMass = reshapedSolution.colwise().mean();
        ycom(i) = centerOfMass(2);
    }

    // Compute acceleration of center of mass
    VectorXd acceleration = (-2 * ycom(seq(1, numSteps - 2)) + ycom.tail(numSteps - 2) + ycom.head(numSteps - 2)) / (dt_ * dt_);
    double averageAcceleration = acceleration(seq(1, numSteps - 4)).mean();

    // Ground truth acceleration
    return (std::abs(averageAcceleration + 10.0) < TOL);
}


bool TestCube::test_pressure_force() {
    // Apply pressure on top surface, elements 1 and 2
    std::vector<std::set<std::string>> elementEnergiesList(elements_.rows());   
    std::vector<Params> elementParameterList(elements_.rows());
    for (int i = 0; i < elements_.rows(); i++) {
        // Insert different energy types for this specific element: neohookean elastic, surface
        elementEnergiesList[i].insert("neohookean");
        // Parameter list
        elementParameterList[i].set_param("density", 1.);
        elementParameterList[i].set_param("youngsModulus", 500000.0);
        elementParameterList[i].set_param("poissonsRatio", 0.4);
    }
    // Define force types (accepted types: "vertexForce", "pressure")
    std::vector<std::string> forceTypesList;
    Params forceParameterList;
    forceTypesList.push_back("pressure");
    forceParameterList.set_param("surfaceVertexIdx", (MatrixXd(2,3) << 1, 5, 6, 1, 6, 2).finished()); // Vertex indices of the surface triangles
    forceParameterList.set_param("surfaceGroups", std::vector<double>{0, 0}); // Group indices of the surface triangles

    // Define actuation signals (for energies, constraints and external forces)
    Params actuation;
    actuation.set_param("pressure", MatrixXd::Zero(1,1)); // Initialize with zeros

    SimulationSettings settings;
    settings.timeSteppingScheme = "backward_euler";
    settings.dt = dt_;
    settings.CFLtimeSteppingFlag = false;
    settings.solverMethod = "minimize_newton";

    Energy<3,4> systemEnergy = Energy<3, 4>(vertices_, elements_, settings, {0.0, 0.0, 0.0}, 0.0, elementEnergiesList, elementParameterList, {}, {}, forceTypesList, forceParameterList);
    Solver<3, 4> solver;

    VectorXd solution = vertices_.reshaped<RowMajor>();
    Matrix<double, -1, 3> reshapedSolution = solution.reshaped<RowMajor>(solution.size() / 3, 3);

    const double maxPressure = 10.0;
    const int numSteps = 10;
    VectorXd ycom = VectorXd::Zero(numSteps);
    ycom(0) = reshapedSolution.colwise().mean()(1);

    for (int i = 1; i < numSteps; i++) {
        MatrixXd act = actuation.get_value("pressure");
        act(0, 0) = maxPressure; 
        actuation.set_param("pressure", act);
        // One step forward in time
        solution = solver.step(solution, systemEnergy, actuation);
        reshapedSolution = solution.reshaped<RowMajor>(solution.size() / 3, 3);
        // Tracking center of mass
        Vector3d centerOfMass = reshapedSolution.colwise().mean();
        ycom(i) = centerOfMass(1);
    }

    // Compute acceleration of center of mass
    VectorXd acceleration = (-2 * ycom(seq(1, numSteps - 2)) + ycom.tail(numSteps - 2) + ycom.head(numSteps - 2)) / (dt_ * dt_);
    double averageAcceleration = acceleration(seq(1, numSteps - 4)).mean();

    // Ground truth acceleration: unit cube has 1kg mass. Force is -10N, acceleration is -10N/1kg = -10m/s^2
    return (std::abs(averageAcceleration + 10.0) < TOL);
}


bool TestCube::test_external_force() {
    std::vector<std::set<std::string>> elementEnergiesList(elements_.rows());   
    std::vector<Params> elementParameterList(elements_.rows());
    for (int i = 0; i < elements_.rows(); i++) {
        elementEnergiesList[i].insert("neohookean");
        // Parameter list
        elementParameterList[i].set_param("density", 1.);
        elementParameterList[i].set_param("youngsModulus", 500000.0);
        elementParameterList[i].set_param("poissonsRatio", 0.4);
    }

    // Define force types (accepted types: "vertexForce", "pressure")
    std::vector<std::string> forceTypesList;
    Params forceParameterList;
    forceTypesList.push_back("vertexForce");

    // Define actuation signals (for energies, constraints and external forces)
    Params actuation;
    actuation.set_param("vertexForce", MatrixXd::Constant(vertices_.rows(), 3, -10.0/vertices_.rows())); // Total force on cube should be 10N, evenly divided on all vertices.

    SimulationSettings settings;
    settings.timeSteppingScheme = "backward_euler";
    settings.dt = dt_;
    settings.CFLtimeSteppingFlag = false;
    settings.solverMethod = "minimize_newton";

    Energy<3,4> systemEnergy = Energy<3, 4>(vertices_, elements_, settings, {0.0, 0.0, 0.0}, 0.0, elementEnergiesList, elementParameterList, {}, {}, forceTypesList, forceParameterList);
    Solver<3, 4> solver;

    VectorXd solution = vertices_.reshaped<RowMajor>();
    // IO::save_tet_VTU("output_test/test_0", vertices_, elements_);
    Matrix<double, -1, 3> reshapedSolution = solution.reshaped<RowMajor>(solution.size()/3, 3);

    const int numSteps = 10;
    VectorXd ycom = VectorXd::Zero(numSteps);
    ycom(0) = reshapedSolution.colwise().mean()(2);
    
    for (int i = 1; i < numSteps; i++) {
        // One step forward in time
        solution = solver.step(solution, systemEnergy, actuation);
        reshapedSolution = solution.reshaped<RowMajor>(solution.size()/3, 3);
        // IO::save_tet_VTU("output_test/test_"+std::to_string(i), reshapedSolution, elements_);
        // Tracking center of mass
        Vector3d centerOfMass = reshapedSolution.colwise().mean();
        ycom(i) = centerOfMass(2);
    }

    // Compute acceleration of center of mass
    VectorXd acceleration = (-2*ycom(seq(1, numSteps-2)) + ycom.tail(numSteps-2) + ycom.head(numSteps-2)) / (dt_*dt_);
    double averageAcceleration = acceleration(seq(1, numSteps-4)).mean();

    // Ground truth acceleration
    return (std::abs(averageAcceleration + 10.0) < TOL);
}


bool TestCube::test_softcon_muscle() {
    // Test the softcon muscle contraction by applying a muscle actuation to several tetrahedra of a cube and verify if the total length decreases.
    std::vector<std::set<std::string>> elementEnergiesList(elements_.rows());
    std::vector<Params> elementParameterList(elements_.rows());
    for (int i = 0; i < elements_.rows(); i++) {
        // Insert different energy types for this specific element: neohookean elastic, softcon muscle
        elementEnergiesList[i].insert("neohookean");

        // Parameter list
        elementParameterList[i].set_param("density", 1000);
        elementParameterList[i].set_param("youngsModulus", 1000000.0);
        elementParameterList[i].set_param("poissonsRatio", 0.4);

        // First two tetrahedra as muscle
        if (i < 2) {
            elementEnergiesList[i].insert("softconMuscle");
            elementParameterList[i].set_param("muscleGroup", i); // set it to the element index number
            elementParameterList[i].set_param("softconStiffness", 100000.0);
            elementParameterList[i].set_param("softconDirection", (MatrixXd(1,3) << 0, 0, 1).finished()); // muscle fiber in z-direction
        }
    }
    Params actuation;
    actuation.set_param("softconMuscle", MatrixXd::Zero(elements_.rows(), 1)); // Initialize with zeros

    SimulationSettings settings;
    settings.timeSteppingScheme = "backward_euler";
    settings.dt = dt_;
    settings.CFLtimeSteppingFlag = false;
    settings.solverMethod = "minimize_newton";

    Energy<3,4> systemEnergy = Energy<3, 4>(vertices_, elements_, settings, {0.0, 0.0, 0.0}, 0.0, elementEnergiesList, elementParameterList);
    Solver<3, 4> solver;

    VectorXd solution = vertices_.reshaped<RowMajor>();
    Matrix<double, -1, 3> reshapedSolution = solution.reshaped<RowMajor>(solution.size() / 3, 3);

    const int numSteps = 10;

    double initialMinZ = reshapedSolution.col(2).minCoeff();
    double initialMaxZ = reshapedSolution.col(2).maxCoeff();
    double initialZlength = initialMaxZ - initialMinZ;

    for (int i = 1; i < numSteps; i++) {
        // Muscle actuation as a function of time
        MatrixXd act = MatrixXd::Zero(elements_.rows(), 1);
        double actuationFactor = 2.0 * i / numSteps;
        for (int j = 0; j < elements_.rows(); j++){
            act(j) = actuationFactor;
        }
        actuation.set_param("softconMuscle", act);
        // One step forward in time
        solution = solver.step(solution, systemEnergy, actuation);
        reshapedSolution = solution.reshaped<RowMajor>(solution.size() / 3, 3);
    }

    // Compute final Z-length
    double finalMinZ = reshapedSolution.col(2).minCoeff();
    double finalMaxZ = reshapedSolution.col(2).maxCoeff();
    double finalZLength = finalMaxZ - finalMinZ;

    // Check if the Z-length has decreased
    return finalZLength < initialZlength;
}


// ---------- TESTING CONSTRAINTS ---------- //
bool TestConstraint::test_plane_contact_no_penetration() {
    // minimal mesh: 1 tetrahedron
    Matrix<double, -1, 3> vertices(4, 3);
    vertices << 0.0, 0.0, -0.10,   // v0 penetrates plane z=0
         1.0, 0.0,  0.20,
         0.0, 1.0,  0.20,
         0.0, 0.0,  0.30;
    Matrix<int, -1, 4> elements(1, 4);
    elements << 0, 1, 2, 3;

    // Set energies and constraints
    std::vector<std::set<std::string>> elementEnergiesList(elements.rows());
    std::vector<Params> elementParameterList(elements.rows());
    for (int i = 0; i < elements.rows(); i++) {
        elementEnergiesList[i].insert("neohookean");
        elementParameterList[i].set_param("density", 1.0);
        elementParameterList[i].set_param("youngsModulus", 1e5);
        elementParameterList[i].set_param("poissonsRatio", 0.3);
    }
    std::vector<std::string> constraintTypesList;
    constraintTypesList.push_back("planeContact");
    Params constraintParams;
    Plane3D plane(Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 1.0)); // z=0, normal +z
    std::vector<Plane3D> planes = {plane};
    std::vector<double> positionVectorListPlanes, normalVectorListPlanes;
    for (const auto& p : planes) {
        for (int k = 0; k < 3; k++) {
            positionVectorListPlanes.push_back(p.get_point()[k]);
            normalVectorListPlanes.push_back(p.get_normal()[k]);
        }
    }

    // For this tiny mesh, treat all vertices as surface vertices
    std::vector<int> surfaceVerticesIdx = {0, 1, 2, 3};

    constraintParams.set_param("positionVectorListPlanes", positionVectorListPlanes);
    constraintParams.set_param("normalVectorListPlanes", normalVectorListPlanes);
    constraintParams.set_param("surfaceVerticesIdx", surfaceVerticesIdx);

    // Simulation settings 
    SimulationSettings settings;
    settings.dt = 1e-2;
    settings.timeSteppingScheme = "backward_euler";
    settings.CFLtimeSteppingFlag = false;
    settings.solverMethod = "minimize_SQP"; // Constraints => SQP

    Energy<3,4> systemEnergy = Energy<3, 4>(vertices, elements, settings, {0.0, 0.0, -9.81}, 0.0,
        elementEnergiesList, elementParameterList, constraintTypesList, constraintParams);
    Solver<3,4> solver;
    Params actuation;

    // Run one constrained step 
    VectorXd x = vertices.reshaped<RowMajor>();
    x = solver.step(x, systemEnergy, actuation);
    Matrix<double, -1, 3> verticesNew = x.reshaped<RowMajor>(x.size()/3, 3);

    // Assert: no penetration (n·(x-p) >= 0) 
    Vector3d n = plane.get_normal();
    Vector3d p0 = plane.get_point();
    for (int vidx : surfaceVerticesIdx) {
        Vector3d vpos = verticesNew.row(vidx).transpose();
        Vector3d vec = vpos - p0;
        double penetration = n.dot(vec);
        if (penetration < -TOL) {
            std::cout << "Vertex " << vidx << " penetrated plane by " << penetration << std::endl;
            return false;
        }
    }
    return true;
}