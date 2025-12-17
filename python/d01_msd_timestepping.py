################################################################################
# Mass–spring–damper simulation demo with gravity and external actuation
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import time
import meshio
import sys
sys.path.append('./cpp/build')
from py_sors import Params, Energy, Solver, IO, SimulationSettings, run_msd_timestepping

################################################################################
# Simulation settings
settings = SimulationSettings()
settings.description = "Demo of a mass-spring damper with external forces under gravity."
settings.meshFilePath = "cpp/demos/meshes/msd_cylinder_v3.msh"
settings.outputFolder = "output"
settings.meshFileName = settings.meshFilePath.split("/")[-1]
settings.meshType = "Tetrahedron"
settings.dt = 1e-2
settings.numSteps = 200
settings.CFLtimeSteppingFlag = True
settings.timeSteppingScheme = "crank_nicolson"
settings.solverMethod = "minimize_newton"
################################################################################

def main(settings):
    """Run a mass–spring–damper demo using a tetrahedral soft-body model.

    Loads the mesh, defines materials, constraints, and external forces,
    advances the simulation using an implicit energy-based solver, and 
    writes results to VTU/PVD files.
    """
    mesh = meshio.read(settings.meshFilePath)

    # Input is in mm, we want m.
    vertices = mesh.points * 1e-3
    elements = None
    for cellBlock in mesh.cells:
        if cellBlock.type == "tetra":
            elements = cellBlock.data
            break
    assert elements is not None, "No tetrahedra found in mesh file"
    minimumZ = np.min(vertices[:, 2])
    maximumZ = np.max(vertices[:, 2])

    # Define what energies need to be used for each element and parameter list.
    elementEnergiesList = []
    elementParameterList = []
    for j in range(elements.shape[0]):
        elementEnergiesList.append(set(["neohookean"]))
        elementParameterList.append(Params(["density", "youngsModulus", "poissonsRatio"], [1070., 12.5*263824.0, 0.499]))

    gravAcceleration = np.array([0.0, 0.0, -9.81])

    # Define constraints
    constraintTypesList = ["neumannBC"]
    # Set boundary conditions
    neumannBCmask = np.zeros(3*vertices.shape[0])
    neumannBCvalue = np.zeros(3*vertices.shape[0])
    for i in range(vertices.shape[0]):
        if vertices[i, 2] > maximumZ - 1e-4:
            neumannBCmask[3*i] = 1;     neumannBCvalue[3*i] = 0.0
            neumannBCmask[3*i+1] = 1;   neumannBCvalue[3*i+1] = 0.0
            neumannBCmask[3*i+2] = 1;   neumannBCvalue[3*i+2] = 0.0
    constraintParameterList = Params(["neumannBCmask", "neumannBCvalue"], [neumannBCmask, neumannBCvalue])

    # Define external forces
    forceTypesList = ["vertexForce"]
    forceParameterList = Params()
    externalVertexForces = np.zeros_like(vertices)
    actuation = Params(["vertexForce"], [externalVertexForces])

    # Initialize energy and solver
    systemEnergy = Energy(vertices, elements, settings, gravAcceleration, 0.0, elementEnergiesList, elementParameterList, constraintTypesList, constraintParameterList, forceTypesList, forceParameterList)
    solver = Solver(settings)

    # Distinguish vertices of cube and rod
    vertexCubeFlag = np.zeros(vertices.shape[0], dtype=bool)
    vertexRodFlag = np.zeros(vertices.shape[0], dtype=bool)
    for i in range(vertices.shape[0]):
        if vertices[i, 2] <= (maximumZ - minimumZ) / 2:
            vertexCubeFlag[i] = True
        else:
            vertexRodFlag[i] = True

    # Initial state
    solution = vertices.flatten()
    IO.save_tet_VTU(f"{settings.outputFolder}/{settings.meshFileName}_0", solution.reshape(-1,3), elements)

    begin = time.time()
    forceMagnitude = 0.5
    for t in range(1, settings.numSteps):
        if t == 105:
            for i in range(vertices.shape[0]):
                if vertexCubeFlag[i]:
                    externalVertexForces[i] = forceMagnitude * np.array([0.01, 0.0, 0.0])
            actuation.set_param("vertexForce", externalVertexForces)
        
        if t == 125:
            externalVertexForces = np.zeros_like(vertices)
            actuation.set_param("vertexForce", externalVertexForces)

        solution = solver.step(solution, systemEnergy, actuation)
        IO.save_tet_VTU(f"{settings.outputFolder}/{settings.meshFileName}_{t}", solution.reshape(-1,3), elements)

        print(f"Time until simulation step {t}: {(time.time()-begin):.2f}s")

    print(f"Average time per simulation step: {1e3*(time.time()-begin)/settings.numSteps:.1f}ms")
    IO.save_pvd(settings.meshFileName, settings.outputFolder, settings.numSteps, settings.dt)

    # startTime = time.time()
    # run_msd_timestepping(meshFile, "output", numSteps)

if __name__ == "__main__":
    settings.verbose_print()

    main(settings)
