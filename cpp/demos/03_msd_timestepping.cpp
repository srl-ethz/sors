#include "demos.h"

/**
 * @brief Run the mass-spring-damper demo with mass-proportional damping and adaptive time stepping.
 *
 * Initializes simulation settings, loads the mass-spring-damper mesh, defines
 * material parameters, gravity, boundary conditions, and time-dependent
 * external forces on the cube, then runs the simulation with CFL-based
 * adaptive time stepping while writing VTU/PVD output and tracking cube
 * displacement and applied force.
 *
 * @param meshFilePath     Path to mesh (.msh).
 * @param outputFolder     Folder where output files are saved.
 * @param numSteps         Number of simulation time steps.
 */
void run_msd_timestepping (std::string meshFilePath, std::string outputFolder, int numSteps) {

    ////////////////////////////////////////////////////////////////////////////////
    // Simulation settings
    SimulationSettings settings;
    settings.description = "Demo of a mass-spring damper with external forces under gravity and adaptive time stepping.";
    settings.meshFilePath = meshFilePath;
    settings.outputFolder = outputFolder;
    settings.meshFileName = meshFilePath.substr(meshFilePath.find_last_of('/') + 1);
    settings.dt = 1e-2; 
    settings.numSteps = numSteps;
    settings.CFLtimeSteppingFlag = true; 
    settings.timeSteppingScheme = "crank_nicolson";
    settings.solverMethod = "minimize_newton"; 
    settings.verbose_print();
    ////////////////////////////////////////////////////////////////////////////////
    
    // Read mesh
    Matrix<double, -1, 3> newVertices;
    Matrix<int, -1, 4> newEleIdx;
    IO::read_tet_MSH(meshFilePath, newVertices, newEleIdx);
    newVertices *= 1e-3; // Input is in mm, we want m.

    // Find minimal and maximal coordinates in x,y,z
    double minimumX, maximumX, minimumY, maximumY, minimumZ, maximumZ;
    minimumX = newVertices.col(0).minCoeff();
    maximumX = newVertices.col(0).maxCoeff();
    minimumY = newVertices.col(1).minCoeff();
    maximumY = newVertices.col(1).maxCoeff();
    minimumZ = newVertices.col(2).minCoeff();
    maximumZ = newVertices.col(2).maxCoeff();

    // Compute center of mass of each element
    Matrix<double, -1, 3> elementCenterOfMass(newEleIdx.rows(), 3);
    for (int i = 0; i < newEleIdx.rows(); i++) {
        Vector<int, 4> vertex_indices = newEleIdx.row(i);
        Vector3d centerOfMass = Vector3d::Zero(); 
        for (int j = 0; j < vertex_indices.size(); j++) {
            centerOfMass += newVertices.row(vertex_indices(j));
        }
        centerOfMass /= vertex_indices.size();
        elementCenterOfMass.row(i) = centerOfMass;
    }

    // Define energies per element and set the required parameters
    std::vector<std::set<std::string>> elementEnergiesList(newEleIdx.rows());
    std::vector<Params> elementParameterList(newEleIdx.rows());
    for (int i = 0; i < newEleIdx.rows(); i++) {
        // Energies 
        elementEnergiesList[i].insert("neohookean");
        // Parameters
        elementParameterList[i].set_param("density", 1070.);
        elementParameterList[i].set_param("youngsModulus", 12.5 * 263824.0);
        elementParameterList[i].set_param("poissonsRatio", 0.499);
    }

    // Define constraint types and set the required parameters 
    std::vector<std::string> constraintTypesList;
    Params constraintParameterList;
    // Constraint types
    constraintTypesList.push_back("neumannBC");
    // Constraint parameters
    std::vector<int> neumannBCmask(3*newVertices.rows(), 0);
    std::vector<double> neumannBCvalue(3*newVertices.rows(), 0);
    for (int i = 0; i < newVertices.rows(); i++) {
        if (newVertices(i, 2) > maximumZ - 1e-4) { // fix the vertices at the top of mass-spring-damper
            neumannBCmask[3*i] = 1; neumannBCvalue[3*i] = 0.0; 
            neumannBCmask[3*i+1] = 1; neumannBCvalue[3*i+1] = 0.0;
            neumannBCmask[3*i+2] = 1; neumannBCvalue[3*i+2] = 0.0;
        }
    }
    // Set active constraints mask for each vertex in 3 dimensions
    constraintParameterList.set_param("neumannBCmask", neumannBCmask);
    // Set neumannBC value (velocity) for each vertex in 3 dimensions
    constraintParameterList.set_param("neumannBCvalue", neumannBCvalue);

    // External forces (time-dependent) for each vertex
    std::vector<std::string> forceTypesList;
    Params forceParameterList;
    forceTypesList.push_back("vertexForce");
    Params actuation;

    Matrix<double, -1, 3> externalVertexForces = Matrix<double, -1, 3>::Zero(newVertices.rows(), 3);
    actuation.set_param("vertexForce", externalVertexForces);

    // Distinguish vertices of cube and rod
    std::vector<int> vertexCubeFlag(newVertices.rows());
    std::vector<int> vertexRodFlag(newVertices.rows());
    for (int i = 0; i < newVertices.rows(); i++) {
        if (newVertices(i, 2) <= (maximumZ-minimumZ)/2) {
            vertexCubeFlag[i] = 1;
            vertexRodFlag[i] = 0;
        }
        else {
            vertexCubeFlag[i] = 0;
            vertexRodFlag[i] = 1;
        }
    }
    
    // Coloring flag for elements
    std::vector<double> elementColorFlag(newEleIdx.rows());;
    for (int i = 0; i < newEleIdx.rows(); i++) {
        if (elementCenterOfMass(i,2) <= (maximumZ-minimumZ)/2) {
            elementColorFlag[i] = 0; // Cube
        }
        else {
            elementColorFlag[i] = 1; // Rod
        }
    }
    VisualOption elementColors;
    elementColors.name = "elementColor";
    elementColors.type = VisualType::ScalarElement;
    elementColors.data = Eigen::RowVectorXd::Map(elementColorFlag.data(), elementColorFlag.size());

    // Tracking CoM of the cube
    // We assume the cube doesnt deform and cube side and rod length is the same.
    // Then, we can just track the vertices at opposite corners of the cube and average them.
    double tolerance = 1e-3;
    int vertexTrackingIdx1 = -1;
    int vertexTrackingIdx2 = -1;
    for (int i = 0; i < newVertices.rows(); i++) {
        if (abs(newVertices(i, 0) - minimumX) < tolerance && 
            abs(newVertices(i, 1) - minimumY) < tolerance && 
            abs(newVertices(i, 2) - minimumZ) < tolerance) {
            vertexTrackingIdx1 = i;
        }
        if (abs(newVertices(i, 0) - maximumX) < tolerance && 
            abs(newVertices(i, 1) - maximumY) < tolerance &&
            abs(newVertices(i, 2) - maximumZ/2.0) < tolerance) {
            vertexTrackingIdx2 = i;
        }
    }
    assert(vertexTrackingIdx1 != -1 && vertexTrackingIdx2 != -1);
    std::vector<double> comHeightTracking(numSteps, 0.0);
    std::vector<double> cubeForceTracking(numSteps, 0.0);
    comHeightTracking[0] = (newVertices.row(vertexTrackingIdx1)(2) + newVertices.row(vertexTrackingIdx2)(2))/2.0;
    cubeForceTracking[0] = 0.0;

    // Initialze system energy and solver
    double massDampingAlpha = 1; // Mass-proportional damping coefficient
    Energy<3, 4> systemEnergy(newVertices, newEleIdx, settings, {0.0, 0.0, -9.81}, massDampingAlpha, elementEnergiesList, elementParameterList, constraintTypesList, constraintParameterList, forceTypesList, forceParameterList);
    Solver<3, 4> solver;

    // Initial state
    VectorXd reshapedVertices = newVertices.reshaped<RowMajor>();
    IO::save_tet_VTU(settings.outputFolder+"/"+settings.meshFileName+"_0", newVertices, newEleIdx, {elementColors});
    VectorXd solution = reshapedVertices;
    Matrix<double, -1, 3> reshapedSolution = solution.reshaped<RowMajor>(solution.size()/3, 3);
    auto begin = std::chrono::steady_clock::now();

    // Run simulation with adaptive time stepping using CFL condition
    double forceMagnitude = 0.5;
    for (int i = 1; i < settings.numSteps; i++) {
        // Apply external forces in x-direction after 105 steps
        if (i == 105) {
            for (int j = 0; j < newVertices.rows(); j++) {
                if (vertexCubeFlag[j] == 1) {
                    externalVertexForces.row(j) = forceMagnitude * Vector3d(0.01, 0.0, 0.0);
                }
            }
            actuation.set_param("vertexForce", externalVertexForces);
        } 
        // Reset external forces after 125 steps
        if (i == 125) {
            actuation.set_param("vertexForce", MatrixXd::Zero(newVertices.rows(), 3));
        }
        // Compute next time step
        solution = solver.step(solution, systemEnergy, actuation);
        // Track points of interest
        reshapedSolution = solution.reshaped<RowMajor>(solution.size()/3, 3);
        comHeightTracking[i] = (reshapedSolution.row(vertexTrackingIdx1)(2) + reshapedSolution.row(vertexTrackingIdx2)(2))/2.0;
        cubeForceTracking[i] = actuation.get_value("vertexForce").row(vertexTrackingIdx1).norm();
        // Save solution
        reshapedSolution = solution.reshaped<RowMajor>(solution.size()/3, 3);
        IO::save_tet_VTU(settings.outputFolder+"/"+settings.meshFileName+"_"+std::to_string(i), reshapedSolution, newEleIdx, {elementColors});
        // Print time
        auto end = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
        std::cout << "Time until simulation step " << i << ": " << elapsed_ms.count() << " ms" << std::endl;
    }

    // Total time
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "Average time per simulation step: " << elapsed_ms.count()/(settings.numSteps-1) << " ms" << std::endl;

    // Save pvd file
    IO::save_pvd(settings.meshFileName, settings.outputFolder, settings.numSteps, settings.dt);

    // Save tracking data 
    std::ofstream file(outputFolder+"/msd_displacement.csv");
    if (!file.is_open()) {
        std::cerr << "Could not open file for writing" << std::endl;
        return;
    } else {
        for (int i = 0; i < settings.numSteps; i++) {
            file << i << "," << i*settings.dt << "," << comHeightTracking[i] << "," << cubeForceTracking[i] << "\n";
        }
        file.close();
    }
   
    std::cout << bcolors.OKBLUE << "Simulation Sucessful" << bcolors.ENDC << std::endl;
    return;
}