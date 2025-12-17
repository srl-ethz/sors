#include "demos.h"

/**
 * @brief Run the bouncing sphere demo on angled planes.
 *
 * Initializes simulation settings, loads the tetrahedral sphere mesh,
 * shifts its initial position, extracts surface vertices for contact
 * handling, assigns Neo-Hookean and gravitational energies, and simulates
 * a soft sphere falling and bouncing between two inclined planes using
 * plane contact constraints. VTU/PVD output is written for visualization.
 *
 * @param meshFilePath     Path to mesh (.msh).
 * @param outputFolder     Folder where output files are saved.
 * @param numSteps         Number of simulation time steps.
 */
void run_bouncing_sphere(std::string meshFilePath, std::string outputFolder, int numSteps) {

    ////////////////////////////////////////////////////////////////////////////////
    // Simulation settings
    SimulationSettings settings;
    settings.description = "Demo of a falling sphere on several angled planes.";
    settings.meshFilePath = meshFilePath;
    settings.outputFolder = outputFolder;
    settings.meshFileName = meshFilePath.substr(meshFilePath.find_last_of('/') + 1);
    settings.dt = 1e-3; 
    settings.numSteps = numSteps;
    settings.CFLtimeSteppingFlag = false; 
    settings.timeSteppingScheme = "crank_nicolson"; 
    settings.solverMethod = "minimize_SQP"; 
    settings.verbose_print();
    ////////////////////////////////////////////////////////////////////////////////

    // Read mesh
    Matrix<double, -1, 3> newVertices;
    Matrix<int, -1, 4> newEleIdx;
    IO::read_tet_MSH(meshFilePath, newVertices, newEleIdx);
    // Transform mesh (shift position of vertices)
    newVertices.col(1) = newVertices.col(1).array() - std::sqrt(0.5)*0.6;
    newVertices.col(2) = newVertices.col(2).array() + 1 - 0.02;

    // Extract the surface vertices
    // Store surface groups, face element groups, and surface vertices 
    auto [surfaceGroups, faceEleGroups] = IO::extract_tet_surfaces(newVertices, newEleIdx);
    std::set<int> surfaceVerticesIdx;
    for (const auto& group : surfaceGroups) {
        for (const auto& face : group) {
            surfaceVerticesIdx.insert(face.begin(), face.end());  // Insert all vertices in the face
        }
    }
    std::vector<int> surfaceVerticesIdxVec(surfaceVerticesIdx.begin(), surfaceVerticesIdx.end());

    // Define energies per element and set the required parameters 
    std::vector<std::set<std::string>> elementEnergiesList(newEleIdx.rows());
    std::vector<Params> elementParameterList(newEleIdx.rows());
    for (int i = 0; i < newEleIdx.rows(); i++) {
        // Energies 
        elementEnergiesList[i].insert("neohookean");
        // Parameters
        elementParameterList[i].set_param("density", 1070.);
        elementParameterList[i].set_param("youngsModulus", 1e7);
        elementParameterList[i].set_param("poissonsRatio", 0.499);
    }

    // Define constraint types and set the required parameters
    std::vector<std::string> constraintTypesList;
    Params constraintParameterList;
    // Constraint types
    constraintTypesList.push_back("planeContact");
    // Constraint parameters
    Plane3D plane1(Vector3d(0.0, std::sqrt(0.5) * 0.5 + 0.01, std::sqrt(0.5) * 0.5 - 0.01), Vector3d(0.0, -1.0, 1));
    Plane3D plane2(Vector3d(0.0, -std::sqrt(0.5) * 0.5 - 0.01, std::sqrt(0.5) * 0.5 - 0.01), Vector3d(0.0, 1.0, 1));
    std::vector<Plane3D> planes = {plane1, plane2};
    std::vector<double> positionVectorListPlanes, normalVectorListPlanes;    
    for (const auto& p : planes) {
        for (int i = 0; i < 3; ++i) {
            positionVectorListPlanes.push_back(p.get_point()[i]);
            normalVectorListPlanes.push_back(p.get_normal()[i]);
        }
    }
    constraintParameterList.set_param("positionVectorListPlanes", positionVectorListPlanes); // Point on the plane
    constraintParameterList.set_param("normalVectorListPlanes", normalVectorListPlanes); // Normal vector of the plane
    constraintParameterList.set_param("surfaceVerticesIdx", surfaceVerticesIdxVec); // Indices of the surface vertices

    // Initialze system energy and solver
    Energy<3, 4> systemEnergy(newVertices, newEleIdx, settings, {0.0, 0.0, -9.81}, 0.0, elementEnergiesList, elementParameterList, constraintTypesList, constraintParameterList);
    Solver<3, 4> solver;

    // Initial state
    VectorXd reshapedVertices = newVertices.reshaped<RowMajor>();
    IO::save_tet_VTU(settings.outputFolder+"/"+settings.meshFileName+"_0", newVertices, newEleIdx, {}, planes);
    VectorXd solution = reshapedVertices;
    Matrix<double, -1, 3> reshapedSolution = solution.reshaped<RowMajor>(solution.size()/3, 3);
    auto begin = std::chrono::steady_clock::now();

    // Run Simulation
    for (int i = 1; i < settings.numSteps; i++) {
        // Compute next time step
        solution = solver.step(solution, systemEnergy);
        // Save solution
        reshapedSolution = solution.reshaped<RowMajor>(solution.size()/3, 3);
        IO::save_tet_VTU(settings.outputFolder+"/"+settings.meshFileName+"_"+std::to_string(i), reshapedSolution, newEleIdx, {}, planes);
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