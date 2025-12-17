#include "demos.h"

/**
 * @brief Run the pressure-actuated SoPrA arm demo.
 *
 * Initializes simulation settings, loads the tetrahedral SoPrA arm mesh,
 * extracts internal pressure chambers, defines material parameters,
 * gravity and fixed-base boundary conditions, and applies time-varying
 * pressure loads to selected chamber groups. Runs the dynamic simulation,
 * writes VTU/PVD output, and logs the arm tip trajectory to CSV.
 *
 * @param meshFilePath     Path to mesh (.msh).
 * @param outputFolder     Folder where output files are saved.
 * @param numSteps         Number of simulation time steps.
 */
void run_sopra_arm (std::string meshFilePath, std::string outputFolder, int numSteps) {
    
    ////////////////////////////////////////////////////////////////////////////////
    // Simulation settings
    SimulationSettings settings;
    settings.description = "Demo of pressure actuated 3D arm (SoPrA Arm).";
    settings.meshFilePath = meshFilePath;
    settings.outputFolder = outputFolder;
    settings.meshFileName = meshFilePath.substr(meshFilePath.find_last_of('/') + 1);
    settings.dt = 1e-2; 
    settings.numSteps = numSteps;
    settings.CFLtimeSteppingFlag = false; 
    settings.timeSteppingScheme = "backward_euler"; 
    settings.solverMethod = "minimize_newton"; 
    settings.verbose_print();
    ////////////////////////////////////////////////////////////////////////////////

    // Read mesh
    Matrix<double, -1, 3> newVertices;
    Matrix<int, -1, 4> newEleIdx;
    IO::read_tet_MSH(meshFilePath, newVertices, newEleIdx);

    // Find minimal and maximal coordinates in x,y,z
    double minimumZ, maximumZ;
    minimumZ = newVertices.col(2).minCoeff();
    maximumZ = newVertices.col(2).maxCoeff();

    // Process surfaces to apply pressure forces
    std::vector<std::vector<std::vector<int>>> surfaceVertexIdx; // [groups][triangle][vertex idx]
    std::vector<std::vector<int>> surfaceEle; // [groups][element idx per triangle]
    std::tie(surfaceVertexIdx, surfaceEle) = IO::extract_tet_surfaces(newVertices, newEleIdx);
    // Remove group with largest size since this will be the outer surface
    int maxGroupIdx = 0;
    std::size_t maxGroupSize = 0;
    for (std::size_t i = 0; i < surfaceVertexIdx.size(); i++) {
        if (surfaceVertexIdx[i].size() > maxGroupSize) {
            maxGroupSize = surfaceVertexIdx[i].size();
            maxGroupIdx = i;
        }
    }
    surfaceVertexIdx.erase(surfaceVertexIdx.begin() + maxGroupIdx);
    surfaceEle.erase(surfaceEle.begin() + maxGroupIdx);
    // Define vector of triangles and their corresponding groups
    std::vector<double> surfaceVertexIdxFlat; // Size: 3*number of surface triangles in all groups
    std::vector<double> surfaceGroupsFlat; // Size: number of surface triangles in all groups
    for (unsigned int groupIdx = 0; groupIdx < surfaceVertexIdx.size(); groupIdx++) {
        const std::vector<std::vector<int>>& triangleList = surfaceVertexIdx[groupIdx];
        if (triangleList.size() == 0) {
            std::cout << "Warning: Surface group " << groupIdx << " has no triangles." << std::endl;
            continue; // Skip empty groups
        }
        for (const auto& triangle : triangleList) {
            if (triangle.size() != 3) {  // Check if the triangle has exactly 3 vertices
                throw std::runtime_error("Each triangle must have exactly 3 vertex indices.");
            }
            for (int idx : triangle) {
                surfaceVertexIdxFlat.push_back(static_cast<double>(idx));
            }
            surfaceGroupsFlat.push_back(static_cast<double>(groupIdx));
        }
    }
    Eigen::Map<Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> svIdx(surfaceVertexIdxFlat.data(), surfaceVertexIdxFlat.size()/3, 3);

    // Define energies per element and set the required parameters
    std::vector<std::set<std::string>> elementEnergiesList(newEleIdx.rows());
    std::vector<Params> elementParameterList(newEleIdx.rows());
    for (int i = 0; i < newEleIdx.rows(); i++) {
        // Energies 
        elementEnergiesList[i].insert("stableneohookean");
        // Parameters
        elementParameterList[i].set_param("density", 1070.);
        elementParameterList[i].set_param("youngsModulus", 263824.0);
        elementParameterList[i].set_param("poissonsRatio", 0.45);
    }

    // Define constraint types and set the required parameters 
    std::vector<std::string> constraintTypesList;
    Params constraintParameterList;
    // Constraint types
    constraintTypesList.push_back("neumannBC");
    // Constraint parameters
    std::vector<int> neumannBCmask(3*newVertices.rows(), 0);
    std::vector<double> neumannBCvalue(3*newVertices.rows(), 0);
    double eps = 1e-4;
    for (int i = 0; i < newVertices.rows(); i++) {
        if (newVertices(i, 2) > maximumZ - eps) { // Fix the vertices at the top of the arm
            neumannBCmask[3*i] = 1; neumannBCvalue[3*i] = 0.0; 
            neumannBCmask[3*i+1] = 1; neumannBCvalue[3*i+1] = 0.0;
            neumannBCmask[3*i+2] = 1; neumannBCvalue[3*i+2] = 0.0;
        }
    }
    // Set active constraints mask for each vertex in 3 dimensions
    constraintParameterList.set_param("neumannBCmask", neumannBCmask);
    // Set neumannBC value (velocity) for each vertex in 3 dimensions
    constraintParameterList.set_param("neumannBCvalue", neumannBCvalue);

    // Define force types 
    std::vector<std::string> forceTypesList;
    Params forceParameterList;
    // Force types
    forceTypesList.push_back("pressure");
    // Force parameters
    forceParameterList.set_param("surfaceVertexIdx", svIdx); // Vertex indices of the surface triangles
    forceParameterList.set_param("surfaceGroups", surfaceGroupsFlat); // Group indices of the surface triangles

    // Define actuation signals (for energies, constraints and external forces)
    Params actuation;
    // Set the actuation parameters for the pressure forces
    int numPressureGroups = static_cast<int>(*max_element(surfaceGroupsFlat.begin(), surfaceGroupsFlat.end())) + 1; // +1 because groups start at 0
    actuation.set_param("pressure", std::vector<double>(numPressureGroups, 0.0)); // Initialize with zeros

    // Average of tip face
    auto get_tip_pos = [](Matrix<double, -1, 3>& vertices, std::vector<int>& tipIdx) {
        Vector3d tipPos = Vector3d::Zero(); 
        for (std::size_t i = 0; i < tipIdx.size(); i++) {
            tipPos += vertices.row(tipIdx[i]);
        }
        tipPos /= tipIdx.size();
        return tipPos;
    };

    // Define maximal pressure for lower and upper segment
    double maxPressureLowerSegment = 80000.0;
    double maxPressureUpperSegment = 50000.0;

    // Initialize energy and solver
    Energy<3, 4> systemEnergy(newVertices, newEleIdx, settings, {0.0, 0.0, -9.81}, 0.0, elementEnergiesList, elementParameterList, constraintTypesList, constraintParameterList, forceTypesList, forceParameterList);
    Solver<3, 4> solver;

    // Track tip index
    std::vector<int> tipIdx;
    for (int i = 0; i < newVertices.rows(); i++) {
        if (newVertices(i, 2) < minimumZ + eps) {
            tipIdx.push_back(i);
        }
    }
    assert(tipIdx.size() > 0);

    // Initial state
    VectorXd reshapedVertices = newVertices.reshaped<RowMajor>();
    IO::save_tet_VTU(settings.outputFolder+"/"+settings.meshFileName+"_0", newVertices, newEleIdx);
    VectorXd solution = reshapedVertices;
    Matrix<double, -1, 3> reshapedSolution = solution.reshaped<RowMajor>(solution.size()/3, 3);
    auto begin = std::chrono::steady_clock::now();
    std::vector<Vector3d> tipPosition(settings.numSteps); // Tracking only the x coordinate.
    tipPosition[0] = get_tip_pos(reshapedSolution, tipIdx);

    // Run simulation 
    for (int i = 1; i < settings.numSteps; i++) {
        // Actuation
        MatrixXd act = actuation.get_value("pressure");
        for (int j = 0; j < numPressureGroups; j++) {
            double pressure = 0.0;
            if (j == 0) {
                pressure = (maxPressureLowerSegment * i/settings.numSteps);
                if (pressure > maxPressureLowerSegment) {pressure = maxPressureLowerSegment;}
            }
            if (j == 3) {
                pressure = (maxPressureUpperSegment * i/settings.numSteps);
                if (pressure > maxPressureUpperSegment) {pressure = maxPressureUpperSegment;}
            }
            act(j, 0) = pressure; // Set the pressure for the current group
        }
        actuation.set_param("pressure", act);
        // Computation of next time step
        solution = solver.step(solution, systemEnergy, actuation);
        // Track points of interest
        reshapedSolution = solution.reshaped<RowMajor>(solution.size()/3, 3);
        tipPosition[i] = get_tip_pos(reshapedSolution, tipIdx);
        // Save solution
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

    // Save tracking data
    std::ofstream file(outputFolder+"/tip_displacement.csv");
    if (!file.is_open()) {
        std::cerr << "Could not open file for writing" << std::endl;
        return;
    } else {
        for (int i = 0; i < numSteps; i++) {
            file << i << "," << i*settings.dt << "," << tipPosition[i][0] << "," << tipPosition[i][1] << "," << tipPosition[i][2] << "\n";
        }
        file.close();
    }

    std::cout << bcolors.OKBLUE << "Simulation Sucessful" << bcolors.ENDC << std::endl;
    return;
}