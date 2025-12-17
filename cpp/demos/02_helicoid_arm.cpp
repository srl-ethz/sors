#include "demos.h"

/**
 * @brief Run the 2-segment helicoid arm tendon actuation demo.
 *
 * Initializes simulation settings, loads the tetrahedral helicoid arm mesh,
 * defines material parameters, boundary conditions, and tendon-like
 * vertex forces on ring segments, then runs the dynamic simulation and
 * writes VTU and PVD output files.
 *
 * @param meshFilePath     Path to mesh (.msh).
 * @param outputFolder     Folder where output files are saved.
 * @param numSteps         Number of simulation time steps.
 */
void run_helicoid_arm (std::string meshFilePath, std::string outputFolder, int numSteps) {
    
    ////////////////////////////////////////////////////////////////////////////////
    // Simulation settings
    SimulationSettings settings;
    settings.description = "Demo of a 2-segement helicoid arm being actuated with tendons.";
    settings.meshFilePath = meshFilePath;
    settings.outputFolder = outputFolder;
    settings.meshFileName = meshFilePath.substr(meshFilePath.find_last_of('/') + 1);
    settings.dt = 1e-2 / 5; 
    settings.numSteps = numSteps;
    settings.CFLtimeSteppingFlag = false; 
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
    double centerX = (minimumX + maximumX) / 2;
    double centerY = (minimumY + maximumY) / 2;
    double centerZ = (minimumZ + maximumZ) / 2;

    // Compute center of mass of each element
    Matrix<double, -1, 3> elementCenterOfMass(newEleIdx.rows(), 3);
    for (unsigned int i = 0; i < newEleIdx.rows(); i++) {
        Vector<int, 4> vertex_indices = newEleIdx.row(i);
        Vector3d centerOfMass = Vector3d::Zero(); 
        for (unsigned int j = 0; j < vertex_indices.size(); j++) {
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
        elementParameterList[i].set_param("density", 1180.0); // data from Flex Semisoft Datasheet
        elementParameterList[i].set_param("youngsModulus", 42*1e6);
        elementParameterList[i].set_param("poissonsRatio", 0.46);
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
        if (newVertices(i, 2) > maximumZ - 1e-4) { // fix the vertices at the top 
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

    // Dividing elements of helicoid arm in subgroups 
    // Ring 1 on top, ring 2 in middle, ring 3 on bottom
    std::vector<int> ring1Flag(newEleIdx.rows(), 0);
    std::vector<int> ring2Flag(newEleIdx.rows(), 0);
    std::vector<int> ring3Flag(newEleIdx.rows(), 0);
    // Elements that are part of the cylinder
    std::vector<int> cylinderFlag(newEleIdx.rows(), 0);
    // Coloring of elements
    std::vector<double> elementColorFlag(newEleIdx.rows());
    for (unsigned int i = 0; i < newEleIdx.rows(); i++) {
        if (elementCenterOfMass(i,2) >= maximumZ - 0.002) {
            elementColorFlag[i] = 0;
            ring1Flag[i] = 1;
        }
        else if (elementCenterOfMass(i,2) <= centerZ + 0.002 && elementCenterOfMass(i,2) >= centerZ - 0.002) {
            elementColorFlag[i] = 0;
            ring2Flag[i] = 1;
        } 
        else if (elementCenterOfMass(i,2) <= minimumZ + 0.002) {
            elementColorFlag[i] = 0;
            ring3Flag[i] = 1;
        } else {
            elementColorFlag[i] = 1;
            cylinderFlag[i] = 1;
        }
    }
    // Find all vertices that are part of the force elements (ring 1,2,3; segment a,b,c,d)
    std::vector<int> ring1aVertices; std::vector<int> ring1bVertices; std::vector<int> ring1cVertices; std::vector<int> ring1dVertices;
    std::vector<int> ring2aVertices; std::vector<int> ring2bVertices; std::vector<int> ring2cVertices; std::vector<int> ring2dVertices;
    std::vector<int> ring3aVertices; std::vector<int> ring3bVertices; std::vector<int> ring3cVertices; std::vector<int> ring3dVertices;
    for (unsigned int i = 0; i < newEleIdx.rows(); i++) {
        // Middle front
        if ((elementCenterOfMass(i,0) >= centerX - 0.003 && elementCenterOfMass(i,0) <= centerX + 0.003 && elementCenterOfMass(i,1) <= centerY)) {
            if (ring1Flag[i] == 1) {
                elementColorFlag[i] = 2;
                ring1aVertices.push_back(newEleIdx(i,0));  ring1aVertices.push_back(newEleIdx(i,1)); 
                ring1aVertices.push_back(newEleIdx(i,2));  ring1aVertices.push_back(newEleIdx(i,3));
            } else if (ring2Flag[i] == 1) {
                elementColorFlag[i] = 2;
                ring2aVertices.push_back(newEleIdx(i,0));  ring2aVertices.push_back(newEleIdx(i,1));
                ring2aVertices.push_back(newEleIdx(i,2));  ring2aVertices.push_back(newEleIdx(i,3));
            } else if (ring3Flag[i] == 1) {
                elementColorFlag[i] = 2;
                ring3aVertices.push_back(newEleIdx(i,0));  ring3aVertices.push_back(newEleIdx(i,1));
                ring3aVertices.push_back(newEleIdx(i,2));  ring3aVertices.push_back(newEleIdx(i,3));
            } else if (cylinderFlag[i] == 1) {
                elementColorFlag[i] = 3;
            }
        } 
        // Right
        else if ((elementCenterOfMass(i,1) >= centerY - 0.003 && elementCenterOfMass(i,1) <= centerY + 0.003 && elementCenterOfMass(i,0) >= centerX)) {
            if (ring1Flag[i] == 1) {
                elementColorFlag[i] = 2;
                ring1bVertices.push_back(newEleIdx(i,0));  ring1bVertices.push_back(newEleIdx(i,1));
                ring1bVertices.push_back(newEleIdx(i,2));  ring1bVertices.push_back(newEleIdx(i,3));
            } else if (ring2Flag[i] == 1) {
                elementColorFlag[i] = 2;
                ring2bVertices.push_back(newEleIdx(i,0));  ring2bVertices.push_back(newEleIdx(i,1));
                ring2bVertices.push_back(newEleIdx(i,2));  ring2bVertices.push_back(newEleIdx(i,3));
            } else if (ring3Flag[i] == 1) {
                elementColorFlag[i] = 2;
                ring3bVertices.push_back(newEleIdx(i,0));  ring3bVertices.push_back(newEleIdx(i,1));
                ring3bVertices.push_back(newEleIdx(i,2));  ring3bVertices.push_back(newEleIdx(i,3));
            } else if (cylinderFlag[i] == 1) {
                elementColorFlag[i] = 3;
            }
        } 
        // Middle back
        else if ((elementCenterOfMass(i,0) >= centerX - 0.003 && elementCenterOfMass(i,0) <= centerX + 0.003 && elementCenterOfMass(i,1) >= centerY)) {
            if (ring1Flag[i] == 1) {
                elementColorFlag[i] = 2;
                ring1cVertices.push_back(newEleIdx(i,0));  ring1cVertices.push_back(newEleIdx(i,1));
                ring1cVertices.push_back(newEleIdx(i,2));  ring1cVertices.push_back(newEleIdx(i,3));
            } else if (ring2Flag[i] == 1) {
                elementColorFlag[i] = 2;
                ring2cVertices.push_back(newEleIdx(i,0));  ring2cVertices.push_back(newEleIdx(i,1));
                ring2cVertices.push_back(newEleIdx(i,2));  ring2cVertices.push_back(newEleIdx(i,3));
            } else if (ring3Flag[i] == 1) {
                elementColorFlag[i] = 2;
                ring3cVertices.push_back(newEleIdx(i,0));  ring3cVertices.push_back(newEleIdx(i,1));
                ring3cVertices.push_back(newEleIdx(i,2));  ring3cVertices.push_back(newEleIdx(i,3));
            } else if (cylinderFlag[i] == 1) {
                elementColorFlag[i] = 3;
            }
        } 
        // Left
        else if ((elementCenterOfMass(i,1) >= centerY - 0.003 && elementCenterOfMass(i,1) <= centerY + 0.003 && elementCenterOfMass(i,0) <= centerX)) {
            if (ring1Flag[i] == 1) {
                elementColorFlag[i] = 2;
                ring1dVertices.push_back(newEleIdx(i,0));  ring1dVertices.push_back(newEleIdx(i,1));
                ring1dVertices.push_back(newEleIdx(i,2));  ring1dVertices.push_back(newEleIdx(i,3));
            } else if (ring2Flag[i] == 1) {
                elementColorFlag[i] = 2;
                ring2dVertices.push_back(newEleIdx(i,0));  ring2dVertices.push_back(newEleIdx(i,1));
                ring2dVertices.push_back(newEleIdx(i,2));  ring2dVertices.push_back(newEleIdx(i,3));
            } else if (ring3Flag[i] == 1) {
                elementColorFlag[i] = 2;
                ring3dVertices.push_back(newEleIdx(i,0));  ring3dVertices.push_back(newEleIdx(i,1));
                ring3dVertices.push_back(newEleIdx(i,2));  ring3dVertices.push_back(newEleIdx(i,3));
            } else if (cylinderFlag[i] == 1) {
                elementColorFlag[i] = 3;
            }
        }
    }
    VisualOption elementColors;
    elementColors.name = "elementColor";
    elementColors.type = VisualType::ScalarElement;
    elementColors.data = Eigen::RowVectorXd::Map(elementColorFlag.data(), elementColorFlag.size());

    // Make a list of all force vertices
    std::vector<std::vector<int>> allForceVertices = {ring1aVertices, ring1bVertices, ring1cVertices, ring1dVertices, 
                                                        ring2aVertices, ring2bVertices, ring2cVertices, ring2dVertices, 
                                                        ring3aVertices, ring3bVertices, ring3cVertices, ring3dVertices}; 
    // Sort and remove duplicates
    for (auto& forceVertices : allForceVertices) {
        std::sort(forceVertices.begin(), forceVertices.end());
        forceVertices.erase(std::unique(forceVertices.begin(), forceVertices.end()), forceVertices.end());
    }
    // We apply forces in a tendon-like manner, by applying forces normal to ring segments of the helicoid arm
    std::vector<int> maxVertexAllRings(12, 0); // Vector with maximal z vertex of each force segment (4 per ring)
    int idxCounter = 0;
    for (auto& forceVertices : allForceVertices) {
        for (unsigned int j = 0; j < forceVertices.size(); j++) {
            if (newVertices(forceVertices[j], 2) > maxVertexAllRings[idxCounter]) {
                maxVertexAllRings[idxCounter] = forceVertices[j];
            }
        }
        idxCounter++;   
    }
    ring1aVertices = allForceVertices[0]; ring1bVertices = allForceVertices[1]; ring1cVertices = allForceVertices[2]; ring1dVertices = allForceVertices[3];
    ring2aVertices = allForceVertices[4]; ring2bVertices = allForceVertices[5]; ring2cVertices = allForceVertices[6]; ring2dVertices = allForceVertices[7];
    ring3aVertices = allForceVertices[8]; ring3bVertices = allForceVertices[9]; ring3cVertices = allForceVertices[10]; ring3dVertices = allForceVertices[11];
    
    // Initialize energy and solver
    double alphaDamping = 500.0;
    Energy<3, 4> systemEnergy(newVertices, newEleIdx, settings, {0.0, 0.0, -9.81}, alphaDamping, elementEnergiesList, elementParameterList, constraintTypesList, constraintParameterList, forceTypesList, forceParameterList);
    Solver<3, 4> solver;

    // Initial state
    VectorXd reshapedVertices = newVertices.reshaped<RowMajor>();
    IO::save_tet_VTU(settings.outputFolder+"/"+settings.meshFileName+"_0", newVertices, newEleIdx, {elementColors});
    VectorXd solution = reshapedVertices;
    Matrix<double, -1, 3> reshapedSolution = solution.reshaped<RowMajor>(solution.size()/3, 3);
    auto begin = std::chrono::steady_clock::now();

    double minForce = 15.0; 
    double maxForce = 50.0;
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    for (int i = 1; i < settings.numSteps; i++) {
        // Find normal vector of plane (upward) of ring 1,2,3 to guide the force vectors with cross product 
        // Vector3d ring1Normal = (reshapedSolution.row(maxVertexAllRings[1]) - reshapedSolution.row(maxVertexAllRings[0])).cross(reshapedSolution.row(maxVertexAllRings[2]) - reshapedSolution.row(maxVertexAllRings[0])).normalized();
        Vector3d ring2Normal = (reshapedSolution.row(maxVertexAllRings[5]) - reshapedSolution.row(maxVertexAllRings[4])).cross(reshapedSolution.row(maxVertexAllRings[6]) - reshapedSolution.row(maxVertexAllRings[4])).normalized();
        Vector3d ring3Normal = (reshapedSolution.row(maxVertexAllRings[9]) - reshapedSolution.row(maxVertexAllRings[8])).cross(reshapedSolution.row(maxVertexAllRings[10]) - reshapedSolution.row(maxVertexAllRings[8])).normalized();

        externalVertexForces =  Matrix<double, -1, 3>::Zero(newVertices.rows(), 3);
        double totalForce2b = 0.0;
        double totalForce4b = 0.0;
        // Add contractile tendon force to the right of the arm on both segments
        // Ring 1 on top, ring 2 in middle, ring 3 on bottom, b is right side of arm
        // Force ramp through tendon forces (lower segement gets half the force of upper segment)
        if (i >= 10 && i < 20) {
            totalForce2b = (minForce + (maxForce - minForce) * (i - 10) / 10.0) * 0.5;
            totalForce4b = minForce + (maxForce - minForce) * (i - 10) / 10.0;
            for (unsigned int j = 0; j < ring2bVertices.size(); j++) {
                externalVertexForces.row(ring2bVertices[j]) += ring2Normal * totalForce2b/ring2bVertices.size();
            }
            for (unsigned int j = 0; j < ring3bVertices.size(); j++) {
                externalVertexForces.row(ring3bVertices[j]) += ring3Normal * totalForce4b/ring3bVertices.size();
            }
        }
        // Force ramp down
        else if (i >= 20 && i < 25) {
            totalForce2b = (maxForce - (maxForce - minForce) * (i - 20) / 5.0) * 0.5;
            totalForce4b = maxForce - (maxForce - minForce) * (i - 20) / 5.0;
            for (unsigned int j = 0; j < ring2bVertices.size(); j++) {
                externalVertexForces.row(ring2bVertices[j]) += ring2Normal * totalForce2b/ring2bVertices.size();
            }
            for (unsigned int j = 0; j < ring3bVertices.size(); j++) {
                externalVertexForces.row(ring3bVertices[j]) += ring3Normal * totalForce4b/ring3bVertices.size();
            }
        }
        actuation.set_param("vertexForce", externalVertexForces);
       
        // Compute next time step
        solution = solver.step(solution, systemEnergy, actuation);
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

    std::cout << bcolors.OKBLUE << "Simulation Sucessful" << bcolors.ENDC << std::endl;
    return;
}