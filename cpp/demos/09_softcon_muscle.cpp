#include "demos.h"
#include <iomanip>

/**
 * @brief Run the SoftCon muscle-on-cylinder demo.
 *
 * Initializes simulation settings, loads the tetrahedral cylinder mesh with
 * embedded SoftCon muscle, assigns Neo-Hookean and SoftCon muscle energies,
 * and applies a linearly increasing muscle actuation signal over time. Runs
 * the dynamic simulation and writes VTU/PVD output for visualization.
 *
 * @param meshFilePath     Path to mesh (.msh).
 * @param outputFolder     Folder where output files are saved.
 * @param numSteps         Number of simulation time steps.
 */
void run_softcon_muscle(std::string meshFilePath, std::string outputFolder, int numSteps)
{
    ////////////////////////////////////////////////////////////////////////////////
    // Simulation settings
    SimulationSettings settings;
    settings.description = "Demo of SoftCon muscle on an actuated cylinder.";
    settings.meshFilePath = meshFilePath;
    settings.outputFolder = outputFolder;
    settings.meshFileName = meshFilePath.substr(meshFilePath.find_last_of('/') + 1);
    settings.dt = 1e-3; 
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
    newVertices *= 1e-3; // Convert from mm to m

    // All flags
    std::vector<double> elementColorFlag(newEleIdx.rows());
    std::vector<int> elementMuscleFlag(newEleIdx.rows());
    for (int i = 0; i < newEleIdx.rows(); i++)
    {
        // Set all elements to muscle
        elementColorFlag[i] = 1;
        elementMuscleFlag[i] = 1;
    };

    // Define energies per element and set the required parameters
    std::vector<std::set<std::string>> elementEnergiesList(newEleIdx.rows());
    std::vector<Params> elementParameterList(newEleIdx.rows());

    // Here we loop over the elements with indexing to access the surfaces
    for (int i = 0; i < newEleIdx.rows(); i++)
    {
        // Define energies per element and set the required parameters
        elementEnergiesList[i].insert("neohookean");
        // Parameter list
        elementParameterList[i].set_param("cpp/demos/parameters/demo_softcon_muscle_param.csv");
        if (elementMuscleFlag[i] != 0)
        {
            elementEnergiesList[i].insert("softconMuscle");
            elementParameterList[i].set_param("muscleGroup", i); // Set it to the element index number
        }
    }

    Params actuation;
    actuation.set_param("softconMuscle", MatrixXd::Zero(elementMuscleFlag.size(), 1)); // Initialize with zeros

    // Initialize energy and solver
    Energy<3, 4> systemEnergy(newVertices, newEleIdx, settings, {0.0, 0.0, 0.0}, 0.0, elementEnergiesList, elementParameterList);
    Solver<3, 4> solver;
    std::vector<std::string> energyVisualizationOption = {"softcon muscle"};

    // initial state
    VectorXd reshapedVertices = newVertices.reshaped<RowMajor>();
    IO::save_tet_VTU(settings.outputFolder+"/"+settings.meshFileName+"_0", newVertices, newEleIdx);
    VectorXd solution = reshapedVertices;
    Matrix<double, -1, 3> reshapedSolution = solution.reshaped<RowMajor>(solution.size() / 3, 3);
    auto begin = std::chrono::steady_clock::now();

    // Run simulation
    for (int i = 1; i < numSteps; i++) {
        // Muscle actuation
        MatrixXd act = MatrixXd::Zero(elementMuscleFlag.size(), 1);
        double actuationFactor = static_cast<double>(i) / numSteps;
        std::cout << "actuationFactor: " << actuationFactor << std::endl;
        for (int j = 0; j < newEleIdx.rows(); j++) {
            if (elementMuscleFlag[j] == 1) {
                act(j) = actuationFactor;
            }
        }
        actuation.set_param("softconMuscle", act);
        // Computation of next time step
        solution = solver.step(solution, systemEnergy, actuation);
        // Save solution
        std::vector<VectorXd> elementEnergy = systemEnergy.compute_elementwise_energy(solution, settings.dt, actuation, energyVisualizationOption);
        reshapedSolution = solution.reshaped<RowMajor>(solution.size() / 3, 3);
        IO::save_tet_VTU(settings.outputFolder+"/"+settings.meshFileName+"_"+std::to_string(i), reshapedSolution, newEleIdx);
        // Print time
        auto end = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
        std::cout << "Time until simulation step " << i << ": " << elapsed_ms.count() << " ms" << std::endl;
    }

    // Total time
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "Average time per simulation step: " << elapsed_ms.count() / (numSteps - 1) << " ms" << std::endl;

    // Save pvd file
    IO::save_pvd(settings.meshFileName, settings.outputFolder, settings.numSteps, settings.dt);

    std::cout << bcolors.OKBLUE << "Simulation Sucessful" << bcolors.ENDC << std::endl;
}