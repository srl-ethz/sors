#include "demos.h"

/**
 * @brief Run the free-floating jelly monkey demo.
 *
 * Initializes simulation settings, loads the tetrahedral monkey mesh, defines
 * material parameters and region-specific time-dependent external forces, and
 * runs the dynamic simulation while writing VTU/PVD output files.
 *
 * @param meshFilePath     Path to mesh (.msh).
 * @param outputFolder     Folder where output files are saved.
 * @param numSteps         Number of simulation time steps.
 */
void run_monkey (std::string meshFilePath, std::string outputFolder, int numSteps) {
    
    ////////////////////////////////////////////////////////////////////////////////
    // Simulation settings
    SimulationSettings settings;
    settings.description = "Demo of a jelly monkey being squeezed with external forces from different directions.";
    settings.meshFilePath = meshFilePath;
    settings.outputFolder = outputFolder;
    settings.meshFileName = meshFilePath.substr(meshFilePath.find_last_of('/') + 1);
    settings.dt = 1e-3; 
    settings.numSteps = numSteps;
    settings.CFLtimeSteppingFlag = false; 
    settings.timeSteppingScheme = "crank_nicolson"; 
    settings.solverMethod = "minimize_newton"; 
    settings.verbose_print();
    ////////////////////////////////////////////////////////////////////////////////
    
    // Read mesh
    Matrix<double, -1, 3> newVertices;
    Matrix<int, -1, 4> newEleIdx;
    IO::read_tet_MSH(settings.meshFilePath, newVertices, newEleIdx);
    newVertices *= 0.01; // Scale to cm scale cube (unit cube length: 1m)

    // Find minimal and maximal coordinates 
    double minimumX, maximumX, minimumZ, maximumZ;
    minimumX = newVertices.col(0).minCoeff(); maximumX = newVertices.col(0).maxCoeff();
    minimumZ = newVertices.col(2).minCoeff(); maximumZ = newVertices.col(2).maxCoeff();

    // Define energies per element and set the required parameters 
    std::vector<std::set<std::string>> elementEnergiesList(newEleIdx.rows());
    std::vector<Params> elementParameterList(newEleIdx.rows());
    for (int i = 0; i < newEleIdx.rows(); i++) {
        // Energies 
        elementEnergiesList[i].insert("stableneohookean");
        // Parameters
        elementParameterList[i].set_param("cpp/demos/parameters/demo_monkey_param.csv");
    }
  
    // External forces (time-dependent) for each vertex
    std::vector<std::string> forceTypesList;
    Params forceParameterList;
    forceTypesList.push_back("vertexForce");
    Params actuation;
    Matrix<double, -1, 3> externalVertexForces = Matrix<double, -1, 3>::Zero(newVertices.rows(), 3);
    actuation.set_param("vertexForce", externalVertexForces);

    // Flag for vertices where we want to apply the external forces
    std::vector<int> externalForceLeftArm(newVertices.rows(), 0);
    std::vector<int> externalForceHead(newVertices.rows(), 0);
    std::vector<int> externalForceRightArm(newVertices.rows(), 0);
    std::vector<int> externalForceLegs(newVertices.rows(), 0);
    for (int i = 0; i < newVertices.rows(); i++) {
        if ((newVertices(i, 0) < minimumX+0.15) && (newVertices(i,2) > maximumZ/2) ){
            externalForceLeftArm[i] = 1;
        }
        if (newVertices(i, 0) > maximumX-0.15){
            externalForceRightArm[i] = 1;
        }
        if ((newVertices(i, 2) > maximumZ*2/3) && (newVertices(i, 0) > minimumX+0.15) && (newVertices(i, 0) < maximumX-0.15)){
            externalForceHead[i] = 1;
        }
        if ((newVertices(i, 2) < minimumZ+0.15) && (newVertices(i, 0) > minimumX+0.15) && (newVertices(i, 0) < maximumX-0.15)){
            externalForceLegs[i] = 1;
        }     
    }

    // Initialze system energy and solver
    double massDampingAlpha = 1.0; 
    Energy<3, 4> systemEnergy(newVertices, newEleIdx, settings, {0.0, 0.0, 0.0}, massDampingAlpha, elementEnergiesList, elementParameterList, {}, {}, forceTypesList, forceParameterList);
    Solver<3, 4> solver;

    // Initial state
    VectorXd reshapedVertices = newVertices.reshaped<RowMajor>();
    IO::save_tet_VTU(settings.outputFolder+"/"+settings.meshFileName+"_0", newVertices, newEleIdx);
    VectorXd solution = reshapedVertices;
    Matrix<double, -1, 3> reshapedSolution = solution.reshaped<RowMajor>(solution.size()/3, 3);
    auto begin = std::chrono::steady_clock::now();

    double forceMagnitude = 0.3;
    // Run simulation
    for (int i = 1; i < settings.numSteps; i++) {
        // Apply external forces after 10 steps
        if (i == 10) {
            for (int j = 0; j < newVertices.rows(); j++) {
                if (externalForceLeftArm[j] == 1)
                    externalVertexForces.row(j) = forceMagnitude * Vector3d(0.03, 0.0, -0.03);
                else if (externalForceRightArm[j] == 1)
                    externalVertexForces.row(j) = forceMagnitude * Vector3d(-0.05, 0.0, 0.0);
                else if (externalForceHead[j] == 1)
                    externalVertexForces.row(j) = forceMagnitude * Vector3d(0.0, 0.0, -0.1);
                else if (externalForceLegs[j] == 1)
                    externalVertexForces.row(j) = forceMagnitude * Vector3d(0.0, 0.0, 0.05);
            }
            actuation.set_param("vertexForce", externalVertexForces);
        }
        // Reset external forces after 50 steps
        if (i == 50) {
            actuation.set_param("vertexForce", MatrixXd::Zero(newVertices.rows(), 3));
        }
        // Compute next time step
        solution = solver.step(solution, systemEnergy, actuation);
        // Save solution
        reshapedSolution = solution.reshaped<RowMajor>(solution.size()/3, 3);
        IO::save_tet_VTU(settings.outputFolder+"/"+settings.meshFileName+"_"+std::to_string(i), reshapedSolution, newEleIdx);
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

    std::cout << bcolors.OKBLUE << "Simulation Sucessful" << bcolors.ENDC << std::endl;
    return;
}