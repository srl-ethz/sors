#ifndef SIMULATIONSETTINGS_H
#define SIMULATIONSETTINGS_H


#include "common.h"

/**
 * @struct SimulationSettings
 * @brief Container for high-level simulation and solver configuration.
 *
 * SimulationSettings groups all run-time options that describe:
 * - the simulation scenario (description, verbosity)
 * - mesh and output paths
 * - time integration settings (dt, numSteps, substeps, CFL control, scheme)
 * - nonlinear solver configuration (solverMethod)
 * - OpenMP parallelization level (numThreads)
 */
struct SimulationSettings {

    // Simulation description
    std::string description;
    int verbose = 1; // Verbosity level, default is 1 (standard information)

    // Mesh and output paths
    std::string meshFilePath;
    std::string outputFolder;
    std::string meshFileName;
    std::string meshType;

    // Time and solver configuration
    double dt = std::numeric_limits<double>::quiet_NaN();  // Uninitialized default
    int numSteps = -1;
    int substeps = 1; // Number of substeps per time step, default is 1 (no substepping)
    bool CFLtimeSteppingFlag = false;
    std::string timeSteppingScheme = "crank_nicolson"; // Default scheme
    std::string solverMethod = "minimize_newton"; // Default solver method
    int numThreads = 1; // Number of threads for parallelization in OpenMP, default is 1 (no parallelization)

    // Verbose output of current settings
    void verbose_print() const {
    std::cout << bcolors.OKBLUE << "\n============================ Simulation Description ============================\n";
    std::cout << (description.empty() ? "not indicated" : description) << bcolors.ENDC << "\n";
    std::cout << "\n============================== Simulation Settings =============================\n";
    std::cout << std::left << std::setw(25) << "Mesh File Path:"
              << (meshFilePath.empty() ? "not indicated" : meshFilePath) << "\n";
    std::cout << std::setw(25) << "Output Folder:"
              << (outputFolder.empty() ? "not indicated" : outputFolder) << "\n";
    std::cout << std::setw(25) << "Mesh File Name:"
              << (meshFileName.empty() ? "not indicated" : meshFileName) << "\n";
    std::cout << std::setw(25) << "Mesh Type"
              << (meshType.empty() ? "not indicated" : meshType) << "\n";
    std::cout << std::setw(25) << "Time Step (dt):";
    if (std::isnan(dt)) {
        std::cout << "not indicated\n";
    } else {
        std::ostringstream dt_str;
        dt_str << dt;
        std::cout << dt_str.str() << "\n";
    }
    std::cout << std::setw(25) << "Number of Steps:"
              << (numSteps < 0 ? "not indicated" : std::to_string(numSteps)) << "\n";
    std::cout << std::setw(25) << "Number of Substeps:" << std::to_string(substeps) << "\n";
    std::cout << std::setw(25) << "CFL Timestepping:"
              << (CFLtimeSteppingFlag ? "On" : "Off") << "\n";
    std::cout << std::setw(25) << "Time Scheme:"
              << (timeSteppingScheme.empty() ? "not indicated" : timeSteppingScheme) << "\n";
    std::cout << std::setw(25) << "Solver Method:"
              << (solverMethod.empty() ? "not indicated" : solverMethod) << "\n";
    std::cout << "================================================================================\n";
}
};

#endif
