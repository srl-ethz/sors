#ifndef DEMOS_H
#define DEMOS_H

#include "energy.h"
#include "solver.h"

// Soft bodies and soft robots
void run_monkey (std::string meshFilePath, std::string outputFolder, int numSteps);
void run_helicoid_arm (std::string meshFilePath, std::string outputFolder, int numSteps);
void run_msd_timestepping (std::string meshFilePath, std::string outputFolder, int numSteps);
void run_sopra_arm (std::string meshFilePath, std::string outputFolder, int numSteps);

// Constraint-based simulations
void run_impact_duck (std::string meshFilePath, std::string outputFolder, int numSteps);
void run_bouncing_sphere (std::string meshFilePath, std::string outputFolder, int numSteps);
void run_squeeze_sphere (std::string meshFilePath, std::string outputFolder, int numSteps);
void run_multiple_constraints_sphere (std::string meshFilePath, std::string outputFolder, int numSteps);

// Softcon Muscle
void run_softcon_muscle (std::string meshFilePath, std::string outputFolder, int numSteps);

// Beam under gravity, hex or tet elements
template<int elementDim>
void run_beam_under_gravity (std::string nameFile, std::string outputFolder, int numSteps);

#endif
