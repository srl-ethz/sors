#include "demos.h"

/**
 * @brief Run the voxel beam under gravity demo.
 *
 * Generates a voxel-based beam mesh (hexahedral or tetrahedral), assigns
 * Neo-Hookean and gravitational energies, fixes one end of the beam via
 * Neumann boundary conditions, and simulates its deformation under gravity.
 * VTU/PVD output is written for visualization.
 *
 * @param meshFilePath     Path to mesh (.msh).
 * @param outputFolder     Folder where output files are saved.
 * @param numSteps         Number of simulation time steps.
 */
template<int elementDim>
void run_beam_under_gravity(std::string filename, std::string outputFolder, int numSteps) {

    ////////////////////////////////////////////////////////////////////////////////
    // Simulation settings
    SimulationSettings settings;
    settings.description = "Demo of a beam under gravity force";
    settings.meshFilePath = "";
    settings.outputFolder = outputFolder;
    settings.meshFileName = filename;
    settings.meshType = (elementDim == 4) ? "Tetrahedron" : "Hexahedron";
    settings.dt = 1e-2; 
    settings.numSteps = numSteps;
    settings.CFLtimeSteppingFlag = false; 
    settings.timeSteppingScheme = "backward_euler"; 
    settings.solverMethod = "minimize_newton"; 
    settings.verbose_print();
    ////////////////////////////////////////////////////////////////////////////////

    // Generate mesh
    Matrix<double, -1, 3> newVertices;
    Matrix<int, -1, elementDim> newEleIdx;
    int n_voxels[3] = {10, 3, 3};
    double edge_dimensions[3] = {0.1, 0.03, 0.03};
    auto voxel_mesh = IO::make_voxel_hex_grid(n_voxels[0], n_voxels[1], n_voxels[2],
                                              edge_dimensions[0] / n_voxels[0], edge_dimensions[1] / n_voxels[1], edge_dimensions[2] / n_voxels[2]);

    newVertices = voxel_mesh.first;
    // Convert Hex elements into Tet elements if elementDim = 4
    if constexpr (elementDim == 4) {
        newEleIdx = IO::hex_to_5_tets(voxel_mesh.second);
    }
    else if constexpr(elementDim == 8) {
        newEleIdx = voxel_mesh.second;
    }
    else {
        throw std::runtime_error("number of vertices can be only 4 or 8");
    }

    // Find minimal and maximal coordinates 
    double minimumX, maximumX, minimumY, maximumY, minimumZ, maximumZ;
    minimumX = newVertices.col(0).minCoeff(); maximumX = newVertices.col(0).maxCoeff();
    minimumY = newVertices.col(1).minCoeff(); maximumY = newVertices.col(1).maxCoeff();
    minimumZ = newVertices.col(2).minCoeff(); maximumZ = newVertices.col(2).maxCoeff();
    UNUSED(minimumX);
    UNUSED(maximumX);
    UNUSED(minimumY);
    UNUSED(maximumY);

    // Extract the surface vertices
    // Store surface groups, face element groups, and surface vertices 
    std::set<int> surfaceVerticesIdx;
    if constexpr (elementDim == 4) {
        auto [surfaceGroups, faceEleGroups] = IO::extract_tet_surfaces(newVertices, newEleIdx);
        for (const auto& group : surfaceGroups) {
            for (const auto& face : group) {
                surfaceVerticesIdx.insert(face.begin(), face.end());  // Insert all vertices in the face
            }
        }
    } else if constexpr (elementDim == 8) {
        auto [surfaceGroups, faceEleGroups] = IO::extract_hex_surfaces(newVertices, newEleIdx);
        for (const auto& group : surfaceGroups) {
            for (const auto& face : group) {
                surfaceVerticesIdx.insert(face.begin(), face.end());  // Insert all vertices in the face
            }
        }
    }
    else {
        throw std::runtime_error("number of vertices can be only 4 or 8");
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
        elementParameterList[i].set_param("youngsModulus", 200e3);
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
    std::vector<int> tipIdx;
    for (int i = 0; i < newVertices.rows(); i++) {
        if (newVertices(i, 0) < minimumX + 1e-4) { 
            neumannBCmask[3*i] = 1; neumannBCvalue[3*i] = 0.0; 
            neumannBCmask[3*i+1] = 1; neumannBCvalue[3*i+1] = 0.0;
            neumannBCmask[3*i+2] = 1; neumannBCvalue[3*i+2] = 0.0;
        }
        if (newVertices(i, 0) > maximumX - 1e-4) {
            tipIdx.push_back(i);
        }
    }
    // Set active constraints mask for each vertex in 3 dimensions
    constraintParameterList.set_param("neumannBCmask", neumannBCmask);
    // Set neumannBC value (velocity) for each vertex in 3 dimensions
    constraintParameterList.set_param("neumannBCvalue", neumannBCvalue);

    // Plane - wall just for visualization
    Plane3D plane1(Vector3d(minimumX, (maximumY+minimumY) / 2., (maximumZ+minimumZ) / 2.), Vector3d(1.0, 0.0, 0.0));
    std::vector<Plane3D> planes = {plane1};

    // Initialze system energy and solver
    Energy<3, elementDim> systemEnergy(newVertices, newEleIdx, settings, {0.0, 0.0, -9.81}, 0.0, elementEnergiesList, elementParameterList, constraintTypesList, constraintParameterList);
    Solver<3, elementDim> solver;

    // Initial state
    VectorXd reshapedVertices = newVertices.reshaped<RowMajor>();
    if constexpr (elementDim == 4) {
        IO::save_tet_VTU(settings.outputFolder+"/"+settings.meshFileName+"_0", newVertices, newEleIdx, {}, planes);
    } else if constexpr (elementDim == 8) {
        IO::save_hex_VTU(settings.outputFolder+"/"+settings.meshFileName+"_0", newVertices, newEleIdx, {}, planes);
    } else {
        throw std::runtime_error("number of vertices can be only 4 or 8");
    }
    VectorXd solution = reshapedVertices;
    Matrix<double, -1, 3> reshapedSolution = solution.reshaped<RowMajor>(solution.size()/3, 3);

    // Average of tip face tracking
    auto get_tip_pos = [](Matrix<double, -1, 3>& vertices, std::vector<int>& tipIdx) {
        Vector3d tipPos = Vector3d::Zero();
        for (std::size_t i = 0; i < tipIdx.size(); i++) {
            tipPos += vertices.row(tipIdx[i]);
        }
        tipPos /= tipIdx.size();
        return tipPos;
    };
    std::vector<Vector3d> tipPosition(settings.numSteps);
    tipPosition[0] = get_tip_pos(reshapedSolution, tipIdx);

    // Run Simulation
    auto begin = std::chrono::steady_clock::now();
    for (int i = 1; i < settings.numSteps; i++) {
        // Compute next time step
        solution = solver.step(solution, systemEnergy);
        // Save solution
        reshapedSolution = solution.reshaped<RowMajor>(solution.size()/3, 3);
        tipPosition[i] = get_tip_pos(reshapedSolution, tipIdx);
        if constexpr (elementDim == 4) {
            IO::save_tet_VTU(settings.outputFolder+"/"+settings.meshFileName+"_"+std::to_string(i), reshapedSolution, newEleIdx, {}, planes);
        }
        else if constexpr (elementDim == 8) {
            IO::save_hex_VTU(settings.outputFolder+"/"+settings.meshFileName+"_"+std::to_string(i), reshapedSolution, newEleIdx, {}, planes);
        }
        else {
            throw std::runtime_error("number of vertices can be only 4 or 8");
        }
        // Print time
        auto end = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
        std::cout << "Time until simulation step " << i << ": " << elapsed_ms.count() << " ms" << std::endl;
    }

    // Total time
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "Average time per simulation step: " << elapsed_ms.count()/(settings.numSteps-1) << " ms" << std::endl;

    // Save tracking data
    std::ofstream file(outputFolder+"/tip_displacement.csv");
    if (!file.is_open()) {
        std::cerr << "Could not open file for writing" << std::endl;
        return;
    } else {
        for (int i = 0; i < settings.numSteps; i++) {
            file << i << "," << i*settings.dt << "," << tipPosition[i][0] << "," << tipPosition[i][1] << "," << tipPosition[i][2] << "\n";
        }
        file.close();
    }

    // Save pvd file
    IO::save_pvd(settings.meshFileName, settings.outputFolder, settings.numSteps, settings.dt);

    std::cout << bcolors.OKBLUE << "Simulation Sucessful" << bcolors.ENDC << std::endl;
    return;
}

template void run_beam_under_gravity<8>(std::string, std::string, int);
template void run_beam_under_gravity<4>(std::string, std::string, int);