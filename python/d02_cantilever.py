################################################################################
# Cantilever Simulation Environment.
#
# Builds a tetrahedral cantilever beam with TetGen, fixes one end via Neumann
# boundary conditions, and simulates passive gravity-driven dynamics.
# Optionally exports VTU/PVD files and offline renders for visualization.
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy
import pyvista as pv
import tetgen
import time
import os
import sys
sys.path.append('./cpp/build')
from py_sors import Params, Energy, Solver, IO, SimulationSettings
from _renderer import PbrtRenderer, export_mp4
plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
mm = 1 / 25.4

################################################################################
# Simulation settings
settings = SimulationSettings()
settings.description = "Demo of passive soft cantilever."
settings.meshFilePath = "cantilever.msh"
settings.outputFolder = "output/cantilever"
settings.meshFileName = settings.meshFilePath.split("/")[-1]
settings.meshType = "Tetrahedron"
settings.dt = 1e-2
settings.numSteps = 100
settings.substeps = 2
settings.CFLtimeSteppingFlag = False
settings.timeSteppingScheme = "crank_nicolson" # "backward_euler", "crank_nicolson"
settings.solverMethod = "minimize_newton"
settings.verbose = 0
settings.numThreads = 8
################################################################################

class BeamEnv:
    """Cantilever beam environment for passive soft-body dynamics.

    Generates a box-shaped tetrahedral mesh (TetGen), pins the base (x-min) using a
    Neumann BC mask/value, defines a Stable Neo-Hookean material per element, The 
    beam tip (x-max) vertex set is tracked for marker-style measurements.
    """
    def __init__ (self, settings, density=1070.0, youngsModulus=263824.0, poissonsRatio=0.49, alpha=50.0):
        # Create output folder
        if not os.path.exists(settings.outputFolder):
            os.makedirs(settings.outputFolder)

        # Create cantilever mesh using TetGen
        # Create a box with dimensions 10cm x 3cm x 3cm
        box = pv.Box(bounds=(0, 10e-2, 0, 3e-2, 0, 3e-2), quads=False, level=3)
        tet = tetgen.TetGen(box)
        tet.tetrahedralize(order=1, mindihedral=20.0, minratio=1.5)
        vertices = np.array(tet.grid.points, dtype=np.float64)
        elements = np.array(tet.grid.cells_dict[10], dtype=np.int32)

        minimumX = np.min(vertices[:, 0])
        maximumX = np.max(vertices[:, 0])

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
        tipIdx = []
        for i in range(vertices.shape[0]):
            if vertices[i, 0] < minimumX + 1e-4:
                neumannBCmask[3*i] = 1;     neumannBCvalue[3*i] = 0.0
                neumannBCmask[3*i+1] = 1;   neumannBCvalue[3*i+1] = 0.0
                neumannBCmask[3*i+2] = 1;   neumannBCvalue[3*i+2] = 0.0
            elif vertices[i, 0] > maximumX - 1e-4:
                tipIdx.append(i)
        constraintParameterList = Params(["neumannBCmask", "neumannBCvalue"], [neumannBCmask, neumannBCvalue])

        # Define external forces
        forceTypesList = ["vertexForce"]
        forceParameterList = Params()

        # Initialize energy and solver
        self.systemEnergy = Energy(vertices, elements, settings, gravAcceleration, alpha, elementEnergiesList, elementParameterList, constraintTypesList, constraintParameterList, forceTypesList, forceParameterList)
        self.solver = Solver(settings)

        self.initialVertices = vertices
        self.elements = elements
        self.tipIdx = tipIdx


    def step (self, solution, actuation=None, dt=0.0):
        if actuation is None:
            actuation = Params(["vertexForce"], [np.zeros(self.initialVertices.shape)])
            return self.solver.step(solution, self.systemEnergy, actuation, dt)
        else:
            return self.solver.step(solution, self.systemEnergy, actuation, dt)
    
    
    def display (self, vertices, markers=None, filename=None, spp=4):
        options = {
            'file_name': filename,
            'light_map': 'uffizi-large.exr',
            'sample': spp,
            'max_depth': 2,
            'camera_pos': (0.2, -0.3, 0.3),   # Position of camera
            'camera_lookat': (-0.05, 0.2, 0.1),     # Position that camera looks at
        }
        transforms=[
                ('s', 1.0),
                ('t', [0, 0, 0.2])
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


def main (settings, density=1070.0, youngsModulus=263824.0, poissonsRatio=0.49, alpha=50, visualize=False):
    """Run the passive cantilever demo and optionally export/plot results.

    Simulates gravity-driven beam motion, tracks the mean tip position over time,
    and produces a z-oscillation plot; when visualize=True, also exports VTU/PVD
    and rendered frames/video.
    """
    # Set up simulation environment, one energy conserving and one dampened.
    sim = BeamEnv(settings, density=density, youngsModulus=youngsModulus, poissonsRatio=poissonsRatio, alpha=alpha)

    ### RUN SIMULATION ###
    solution = sim.initialVertices.copy().flatten()
    if visualize:
        IO.save_tet_VTU(f"{settings.outputFolder}/{settings.meshFileName}_0", solution.reshape(-1,3), sim.elements)
        sim.display(solution.reshape(-1,3), filename=f"{settings.outputFolder}/{settings.meshFileName}_{0:04d}.png", spp=4)

    begin = time.time()
    markerPos = [solution.reshape(-1,3)[sim.tipIdx].mean(0)]
    for t in range(1, settings.numSteps):
        solution = sim.step(solution)

        if visualize:
            IO.save_tet_VTU(f"{settings.outputFolder}/{settings.meshFileName}_{t}", solution.reshape(-1,3), sim.elements)
            sim.display(solution.reshape(-1,3), filename=f"{settings.outputFolder}/{settings.meshFileName}_{t:04d}.png", spp=4)

        markerPos.append(solution.reshape(-1,3)[sim.tipIdx].mean(0))
        print(f"Time until simulation step {t}: {(time.time()-begin):.2f}s")
    
    markerPos = np.stack(markerPos, axis=0)
    print(f"Average time per simulation step: {1e3*(time.time()-begin)/settings.numSteps:.1f}ms with total time {time.time()-begin:.2f}s")

    if visualize:
        IO.save_pvd(settings.meshFileName, settings.outputFolder, settings.numSteps, settings.dt)
        export_mp4(f"{settings.outputFolder}", f"{settings.meshFileName}.mp4", int(1.0/settings.dt))

    if not os.path.exists(f"{settings.outputFolder}/plots"):
        os.makedirs(f"{settings.outputFolder}/plots")

    # Plot marker oscillations
    timeAxis = np.arange(len(markerPos)) * settings.dt
    fig, ax = plt.subplots(figsize=(40*mm, 30*mm))
    ax.plot(timeAxis, markerPos[:,2])
    ax.grid()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Z Position (m)")
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
    fig.savefig(f"{settings.outputFolder}/plots/oscillations.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # settings.verbose_print()
    main(settings, density=1070, youngsModulus=263824, poissonsRatio=0.49, alpha=50, visualize=True)
