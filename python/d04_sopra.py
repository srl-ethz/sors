################################################################################
# SoPrA pressure-actuated soft arm demo
#
# Loads a tetrahedral arm mesh, applies internal pressure actuation, advances the
# dynamics with an implicit FEM solver, and optionally renders and exports a video.
# ################################################################################

import os
import sys
import time
import meshio
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('./cpp/build')
from py_sors import Params, Energy, Solver, IO, SimulationSettings
from _renderer import PbrtRenderer, export_mp4

################################################################################
# Simulation settings
settings = SimulationSettings()
settings.description = "Demo of pressure actuated 3D arm (SoPrA Arm)."
settings.meshFilePath = "cpp/demos/meshes/sopra.msh"
settings.outputFolder = "output/sopra"
settings.meshFileName = settings.meshFilePath.split("/")[-1]
settings.meshType = "Tetrahedron"
settings.dt = 1e-1
settings.numSteps = 25
settings.substeps = 10
settings.CFLtimeSteppingFlag = False
settings.timeSteppingScheme = "backward_euler"
settings.solverMethod = "minimize_newton"
settings.verbose = 0
settings.numThreads = 8
################################################################################

class SopraEnv:
    """Simulation wrapper for the SoPrA pressure-actuated soft arm.

    Handles mesh loading, surface extraction for pressure chambers, boundary
    conditions, energy/solver setup, and provides simple step and render helpers.
    """
    
    def __init__ (self, settings, density=1070.0, youngsModulus=263824.0, poissonsRatio=0.49, alpha=50.0):
        # Create output folder
        if not os.path.exists(settings.outputFolder):
            os.makedirs(settings.outputFolder)

        # Read Mesh
        mesh = meshio.read(settings.meshFilePath)
        vertices = mesh.points
        elements = None
        for cellBlock in mesh.cells:
            if cellBlock.type == "tetra":
                elements = cellBlock.data
                break
        assert elements is not None, "No tetrahedra found in mesh file"
        minimumZ = np.min(vertices[:, 2])
        maximumZ = np.max(vertices[:, 2])
        # Center base at origin and track tip indices
        baseIdx = []
        self.tipIdx = []
        for i in range(vertices.shape[0]):
            if vertices[i, 2] > maximumZ - 1e-4:
                baseIdx.append(i)
            elif vertices[i, 2] < minimumZ + 1e-4:
                self.tipIdx.append(i)
        baseCenter = np.mean(vertices[baseIdx], axis=0)
        vertices -= baseCenter
        vertices += np.array([0., 0., 0.05]) # Lift up a bit for visualization
        minimumZ = np.min(vertices[:, 2])
        maximumZ = np.max(vertices[:, 2])

        # Process surfaces to apply pressure forces
        surfaceVertexIdx, _ = IO.extract_tet_surfaces(vertices, elements)
        # Remove group with largest size since this will be the outer surface
        maxGroupIdx = 0
        maxGroupSize = 0
        for i in range(len(surfaceVertexIdx)):
            if len(surfaceVertexIdx[i]) > maxGroupSize:
                maxGroupSize = len(surfaceVertexIdx[i])
                maxGroupIdx = i
        surfaceVertexIdx.pop(maxGroupIdx)
        # Flatten list of triangles
        surfaceGroups = np.concatenate([i*np.ones(len(surfaceVertexIdx[i])) for i in range(len(surfaceVertexIdx))])
        surfaceVertexIdx = np.concatenate([np.array(surfaceVertexIdx[i]) for i in range(len(surfaceVertexIdx))])

        # Define what energies need to be used for each element and parameter list.
        elementEnergiesList = []
        elementParameterList = []
        for j in range(elements.shape[0]):
            elementEnergiesList.append(set(["stableneohookean"]))
            elementParameterList.append(Params(["density", "youngsModulus", "poissonsRatio"], [density, youngsModulus, poissonsRatio]))

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
        forceTypesList = ["pressure"]
        forceParameterList = Params(["surfaceVertexIdx", "surfaceGroups"], [surfaceVertexIdx, surfaceGroups])

        # Initialize energy and solver
        self.systemEnergy = Energy(vertices, elements, settings, gravAcceleration, alpha, elementEnergiesList, elementParameterList, constraintTypesList, constraintParameterList, forceTypesList, forceParameterList)
        self.solver = Solver(settings)

        self.initialVertices = vertices
        self.elements = elements


    def step (self, solution, actuation):
        """Advance the simulation by one time step.

        Uses the internal SORS solver to update vertex positions given the
        current state and pressure actuation.
        """
        return self.solver.step(solution, self.systemEnergy, actuation)
    
    
    def display (self, vertices, markers=None, filename=None, spp=4):
        """Render the current arm configuration using PBRT.

        Optionally saves a rendered image and visualizes markers such as tip points.
        """
        options = {
            'file_name': filename,
            'light_map': 'uffizi-large.exr',
            'sample': spp,
            'max_depth': 2,
            'camera_pos': (0, -0.75, 0.5),   # Position of camera
            'camera_lookat': (0., 0.25, 0.),     # Position that camera looks at
            # 'camera_pos': (0, -0.01, 0.5),   # Position of camera
            # 'camera_lookat': (0., 0., 0.),     # Position that camera looks at
        }
        transforms=[
                ('s', 1.0),
                ('t', [0, 0, 0.3])
        ]
        renderer = PbrtRenderer(options)

        renderer.add_tri_mesh(vertices=vertices, elements=self.elements, render_edges=True, color="496d8a", transforms=transforms)

        # Add markers as spheres
        if markers is not None:
            colors = ['ff7f0e', '2ca02c', 'd62728', '9467bd', '8c564b', 'e377c2', '7f7f7f', 'bcbd22', '17becf'] # orange, green, red, purple, brown, pink, gray, yellow-green, cyan
            for i in range(markers.shape[0]):
                renderer.add_shape_mesh({'name': 'sphere', 'center': markers[i], 'radius': 0.0075}, color=colors[i%9], transforms=transforms)

        renderer.add_tri_mesh(objFile='asset/mesh/curved_ground.obj', texture_img='chkbd_24_0.7', transforms=[('s', 4)])
        renderer.render()


def main(settings, visualize=False):
    """Run the SoPrA arm demo with ramped pressure actuation.

    Initializes the environment, applies time-varying chamber pressures, advances
    the simulation, writes VTU/PVD output, and optionally renders a video.
    """
    # Initialize simulation
    sim = SopraEnv(settings, density=1070, youngsModulus=263824, poissonsRatio=0.49, alpha=50)
    actuation = Params(["pressure"], [np.zeros([6,1])]) # 6 chambers
    
    # Define maximal pressure for lower and upper segment
    maxPressureLowerSegment = 50000
    maxPressureUpperSegment = 30000

    # Initial state
    solution = sim.initialVertices.copy().flatten()
    if visualize:
        IO.save_tet_VTU(f"{settings.outputFolder}/{settings.meshFileName}_0", solution.reshape(-1,3), sim.elements)
        sim.display(solution.reshape(-1,3), filename=f"{settings.outputFolder}/{settings.meshFileName}_{0:04d}.png", spp=4)

    # Run simulation with adaptive time stepping using CFL condition
    begin = time.time()
    for t in range(1, settings.numSteps):
        # Actuation
        act = actuation.get_value("pressure")
        act[0,0] = maxPressureLowerSegment * t/settings.numSteps
        act[3,0] = maxPressureUpperSegment * t/settings.numSteps
        actuation.set_param("pressure", act)

        solution = sim.step(solution, actuation)

        if visualize:
            IO.save_tet_VTU(f"{settings.outputFolder}/{settings.meshFileName}_{t}", solution.reshape(-1,3), sim.elements)
            sim.display(solution.reshape(-1,3), filename=f"{settings.outputFolder}/{settings.meshFileName}_{t:04d}.png", spp=4)
        print(f"Time until step {t}: {(time.time()-begin):.2f}s, Pressures: {actuation.get_value('pressure').flatten()}")

    print(f"Average time per simulation step: {1e3*(time.time()-begin)/settings.numSteps:.1f}ms")
    IO.save_pvd(settings.meshFileName, settings.outputFolder, settings.numSteps, settings.dt)

    if visualize:
        export_mp4(f"{settings.outputFolder}", f"{settings.meshFileName}.mp4", int(1.0/settings.dt))


if __name__ == "__main__":
    settings.verbose_print()
    main(settings, visualize=True)
