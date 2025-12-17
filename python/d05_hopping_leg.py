################################################################################
# Hopping-leg default environment (hex or tet).
#
# Builds a simple voxelized leg, converts it to a hex mesh (or to tets via
# `IO.hex_to_5_tets`), assigns Neo-Hookean elasticity plus optional muscle groups,
# and simulates contact against a ground plane using `planeContact`. Includes a
# lightweight actuation schedule helper and a demo `main()` that logs COM / min-Z,
# writes VTU/PVD outputs, renders frames, and produces plots for trajectory and
# actuation.
################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import time, sys
sys.path.append('./cpp/build')
from py_sors import Params, Energy, Solver, IO, SimulationSettings, Plane3D, VisualOption, VisualType
from _renderer import PbrtRenderer, export_mp4, filter_elements
plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
mm = 1 / 25.4

################################################################################
# Simulation settings
settings = SimulationSettings()
settings.description = "Demo of a hopping leg that is muscle-actuated."
settings.meshFilePath = "hopping_leg.msh"
settings.outputFolder = "output/hopping_leg"
settings.meshFileName = settings.meshFilePath.split("/")[-1]
settings.meshType = "Hexahedron"
settings.dt = 2e-2
settings.numSteps = 75
settings.substeps = 25
settings.CFLtimeSteppingFlag = False
settings.timeSteppingScheme = "backward_euler" # "backward_euler", "crank_nicolson"
settings.solverMethod = "minimize_SQP"
settings.verbose = 0
settings.numThreads = 4
################################################################################

def voxel2hex (voxels, dx, dy, dz):
    """Convert a boolean voxel grid into a hexahedral mesh.

    Creates one vertex per active voxel corner (deduplicated across neighboring
    voxels) and builds hex elements using the library-specific vertex ordering
    expected by IO (see io.cpp face ordering).

    Args:
        voxels: (nx, ny, nz) array-like of bool/int, where True/1 marks an active cell.
        dx: Cell size along X (meters).
        dy: Cell size along Y (meters).
        dz: Cell size along Z (meters).

    Returns:
        vertices: (N, 3) float array of vertex positions.
        elements: (E, 8) int array of hexahedron indices.
    """

    nx, ny, nz = voxels.shape # Number of cells
    # One more vertex than cells in each direction
    vertexFlag = np.full((nx+1, ny+1, nz+1), -1, dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if voxels[i][j][k]:
                    for ii in range(2):
                        for jj in range(2):
                            for kk in range(2):
                                vertexFlag[i + ii, j + jj, k + kk] = 0

    vertexCnt = 0
    vertices = []
    for i in range(nx+1):
        for j in range(ny+1):
            for k in range(nz+1):
                if vertexFlag[i,j,k] == 0:
                    vertexFlag[i,j,k] = vertexCnt
                    vertices.append((dx * i, dy * j, dz * k))
                    vertexCnt += 1

    # Very specific hexahedron face ordering based on io.cpp.
    elements = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if voxels[i,j,k]:
                    elements.append([
                        vertexFlag[i,j,k],
                        vertexFlag[i,j+1,k],
                        vertexFlag[i+1,j+1,k],
                        vertexFlag[i+1,j,k],
                        vertexFlag[i,j,k+1],
                        vertexFlag[i,j+1,k+1],
                        vertexFlag[i+1,j+1,k+1],
                        vertexFlag[i+1,j,k+1],
                    ])

    vertices = np.stack(vertices, axis=0)
    elements = np.stack(elements, axis=0)

    return vertices, elements


class MuscleEnv:
    """Default hopping-leg simulation environment with ground contact.

    Generates a simple “upper-leg + foot” voxel shape, converts it into a hex mesh
    (or tets if `settings.meshType == "Tetrahedron"`), and builds an
    `Energy` + `Solver` system with gravity, optional muscle activation, and
    plane contact against z=0.
    """
    def __init__ (self, settings):
        # Create output folder
        if not os.path.exists(settings.outputFolder):
            os.makedirs(settings.outputFolder)

        # Create Leg Mesh
        nx, ny, nz = 5, 4, 10
        voxel = np.zeros((nx, ny, nz))
        # Upper Leg
        voxel[0:2, 1:3, 2:10] = 1
        # Lower Foot
        voxel[0:5, 0:4, 0:2] = 1
        # Generate Mesh
        vertices, elements = voxel2hex(voxel, 0.05, 0.05, 0.05)
        vertices += np.array([0.0, -0.1, 1.0])
        if settings.meshType == "Tetrahedron":
            elements = IO.hex_to_5_tets(elements)

        minimumZ = np.min(vertices[:, 2])

        # Define what energies need to be used for each element and parameter list.
        elementEnergiesList = []
        elementParameterList = []
        muscleEleIdx = {
            "0": [],
            "1": [],
        }
        for i in range(elements.shape[0]):
            energyList = set(["neohookean"])
            parameterList = Params(["density", "youngsModulus", "poissonsRatio"], [1000., 500000.0, 0.4])

            ### Add Muscle Units
            eleCom = np.mean(vertices[elements[i]], axis=0)
            if eleCom[2] > minimumZ+0.15 and eleCom[2] < minimumZ+0.4:
                energyList.add("softconMuscle")
                parameterList.set_param("softconStiffness", 500000.0)
                parameterList.set_param("softconDirection", [0, 0, 1])
                
                if eleCom[0] < 0.05:
                    muscleEleIdx["0"].append(i)
                    parameterList.set_param("muscleGroup", 0)
                elif eleCom[0] >= 0.05:
                    muscleEleIdx["1"].append(i)
                    parameterList.set_param("muscleGroup", 1)

            elementEnergiesList.append(energyList)
            elementParameterList.append(parameterList)

        gravAcceleration = np.array([0.0, 0.0, -9.81])

        # Define constraints
        constraintTypesList = ["planeContact", "neumannBC"]
        neumannBCmask = np.zeros(3*vertices.shape[0])
        neumannBCvalue = np.zeros(3*vertices.shape[0])
        self.baseIdx = []
        for i in range(vertices.shape[0]):
            if vertices[i, 2] <= minimumZ+1e-3:
                self.baseIdx.append(i)
        # Define plane contact
        planes = [Plane3D(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))]
        positionVectorListPlanes = []
        normalVectorListPlanes = []
        for plane in planes:
            positionVectorListPlanes.append(plane.get_point())
            normalVectorListPlanes.append(plane.get_normal())
        if settings.meshType == "Tetrahedron":
            surfaceGroups, faceEleGroups = IO.extract_tet_surfaces(vertices, elements)
        elif settings.meshType == "Hexahedron":
            surfaceGroups, faceEleGroups = IO.extract_hex_surfaces(vertices, elements)
        surfaceVerticesIdx = np.unique(surfaceGroups)

        constraintParameterList = Params(["neumannBCmask", "neumannBCvalue", "positionVectorListPlanes", "normalVectorListPlanes", "surfaceVerticesIdx"], [neumannBCmask, neumannBCvalue, positionVectorListPlanes, normalVectorListPlanes, surfaceVerticesIdx])

        # Define external forces
        forceTypesList = ["vertexForce"]
        forceParameterList = Params()

        # Initialize energy and solver
        self.systemEnergy = Energy(vertices, elements, settings, gravAcceleration, 0.0, elementEnergiesList, elementParameterList, constraintTypesList, constraintParameterList, forceTypesList, forceParameterList)
        self.solver = Solver(settings)

        initialVelocity = np.ones_like(vertices) * 0.5
        self.systemEnergy.set_initial_velocity(initialVelocity)

        self.initialVertices = vertices
        self.elements = elements

        self.muscleEleIdx = muscleEleIdx

        # Element color flags
        self.elementColorFlag = np.zeros(elements.shape[0])
        for i, group in enumerate(self.muscleEleIdx):
            for idx in self.muscleEleIdx[group]:
                self.elementColorFlag[idx] = i+1
                
        self.visuals = []
        visualOption = VisualOption()
        visualOption.name = "elementColor"
        visualOption.type = VisualType.ScalarElement
        visualOption.data = self.elementColorFlag.reshape(1, -1).astype(float)
        self.visuals.append(visualOption)
        

    def step (self, solution, actuation):
        return self.solver.step(solution, self.systemEnergy, actuation)
    
    
    def display (self, vertices, filename=None, spp=4):
        options = {
            'file_name': filename,
            'light_map': 'uffizi-large.exr',
            'sample': spp,
            'max_depth': 2,
            'camera_pos': (0.5, -3.5, 1.25),   # Position of camera
            'camera_lookat': (0.5, 0, .5)     # Position that camera looks at
        }
        transforms=[
                ('s', 1.0),
                ('t', [-0.05, -0.05, 0.0])
        ]
        renderer = PbrtRenderer(options)

        structureVertices, structureElements = filter_elements(vertices, self.elements, [i for i in range(self.elements.shape[0]) if (i not in self.muscleEleIdx["0"] and i not in self.muscleEleIdx["1"])])
        if settings.meshType == "Tetrahedron":
            renderer.add_tri_mesh(vertices=structureVertices, elements=structureElements, render_edges=True, color="496d8a", transforms=transforms)
        elif settings.meshType == "Hexahedron":
            renderer.add_hex_mesh(vertices=structureVertices, elements=structureElements, render_edges=True, color="496d8a", transforms=transforms)

        colors = ["ff684a", "ffb74a"]
        for i, group in enumerate(self.muscleEleIdx):
            muscleVertices, muscleElements = filter_elements(vertices, self.elements, self.muscleEleIdx[group])
            if settings.meshType == "Tetrahedron":
                renderer.add_tri_mesh(vertices=muscleVertices, elements=muscleElements, render_edges=True, color=colors[i], transforms=transforms)
            elif settings.meshType == "Hexahedron":
                renderer.add_hex_mesh(vertices=muscleVertices, elements=muscleElements, render_edges=True, color=colors[i], transforms=transforms)

        renderer.add_tri_mesh(objFile='asset/mesh/curved_ground.obj', texture_img='chkbd_24_0.7', transforms=[('s', 4)])
        renderer.render()


class Actuation:
    """Piecewise actuation schedule with smooth ramps.

    Implements a simple timing model: start from 0, ramp to `startVal`, hold, then
    transition to `endVal` around `stepTime`. Works with scalar or per-channel
    parameters (vectorized via arrays).

    Args:
        stepTime: Time(s) at which the actuation switches from startVal→endVal.
            Scalar or sequence (one per channel).
        startVal: Initial plateau value(s) after the initial ramp-up.
        endVal: Final plateau value(s) after the step transition.
    """
    def __init__ (self, stepTime, startVal, endVal):
        self.startVal = np.array(startVal) if isinstance(startVal, (list, np.ndarray)) else np.array([startVal])
        self.endVal = np.array(endVal) if isinstance(endVal, (list, np.ndarray)) else np.array([endVal])
        self.stepTime = np.array(stepTime) if isinstance(stepTime, (list, np.ndarray)) else np.array([stepTime]*len(self.startVal))
        assert len(self.startVal) == len(self.endVal) == len(self.stepTime), "startVal, endVal and stepTime must be of the same length."
        slope = 15
        self.initialTransition = abs(self.startVal) / slope
        self.transitionDuration = abs(self.endVal-self.startVal) / slope


    def __call__ (self, t):
        # Actuation always starts at zero, ramps up to startVal in transitionDuration, then steps to endVal at stepTime in transitionDuration.
        if isinstance(t, np.ndarray):
            return np.stack([self(ti) for ti in t], axis=0)
        else:
            result = []
            for i in range(len(self.startVal)):
                if t < self.initialTransition[i]:
                    result.append((t/self.initialTransition[i])*self.startVal[i])
                elif t < self.stepTime[i] - self.transitionDuration[i]/2:
                    result.append(self.startVal[i])
                elif t < self.stepTime[i] + self.transitionDuration[i]/2:
                    result.append(((t-(self.stepTime[i]-self.transitionDuration[i]/2))/self.transitionDuration[i])*(self.endVal[i]-self.startVal[i]) + self.startVal[i])
                else:
                    result.append(self.endVal[i])
            return np.stack(result, axis=0)
        

def main (settings, stepTime=[0.5, 0.5], startVal=[0.0, 0.0], endVal=[0.0, 0.0], visualize=False):
    """Run a hopping-leg rollout and log jump/trajectory metrics.

    Builds a `MuscleEnv`, drives a two-channel `softconMuscle` actuation via the
    `Actuation` helper, and simulates plane contact under gravity. Logs:
      - min-Z vertex position over time (proxy for ground clearance / jump height),
      - center-of-mass trajectory,
      - actuation signal,
      - distance traveled in +X and max height after 0.5s.

    Also saves CSV logs and plots, and optionally writes VTU/PVD + renders frames.

    Args:
        settings: SimulationSettings instance (dt/numSteps/substeps/solver/output paths).
        stepTime: Sequence of length 2, step time(s) for each muscle group (seconds).
        startVal: Sequence of length 2, pre-step activation levels.
        endVal: Sequence of length 2, post-step activation levels.
        visualize: If True, saves VTU/PVD outputs and renders frames + MP4.

    Returns:
        log: Dict with keys:
            - "minZ": (T,3) min-z vertex positions over time.
            - "com": (T,3) center-of-mass over time.
            - "actSignal": (T,2) actuation signal over time.
            - "distance": float, COM x-displacement from start to end.
            - "maxPos": float, max height metric (after 0.5s window if available).
    """
    
    if not os.path.exists(f"{settings.outputFolder}/opt_results"):
        os.makedirs(f"{settings.outputFolder}/opt_results")
    if not os.path.exists(f"{settings.outputFolder}/plots"):
        os.makedirs(f"{settings.outputFolder}/plots")

    sim = MuscleEnv(settings)
    externalVertexForces = np.zeros_like(sim.initialVertices)
    actuation = Params(["softconMuscle", "vertexForce"], [np.zeros([2,1]), externalVertexForces])

    # Initial state
    solution = sim.initialVertices.flatten()
    if visualize:
        if settings.meshType == "Tetrahedron":
            IO.save_tet_VTU(f"{settings.outputFolder}/{settings.meshFileName}_0", solution.reshape(-1,3), sim.elements, sim.visuals)
        elif settings.meshType == "Hexahedron":
            IO.save_hex_VTU(f"{settings.outputFolder}/{settings.meshFileName}_0", solution.reshape(-1,3), sim.elements, sim.visuals)

    # Actuation
    timeAxis = np.arange(settings.numSteps)*settings.dt
    actSignal = Actuation(stepTime=stepTime, startVal=startVal, endVal=endVal)
    # Store actuation signal
    np.savetxt(f"{settings.outputFolder}/plots/actuationSignal.csv", np.concatenate([timeAxis.reshape(-1,1), actSignal(timeAxis)], axis=-1), delimiter=',')

    # Run simulation
    minIdx = np.argmin(solution.reshape(-1,3)[:,2])
    log = {
        "minZ": [solution.reshape(-1,3)[minIdx]],
        "com": [solution.reshape(-1,3).mean(axis=0)],
        "actSignal": actSignal(timeAxis),
        "distance": 0.0,
        "maxPos": 0.0,
    }
    begin = time.time()
    for t in range(1, settings.numSteps):
        actuation.set_param("softconMuscle", actSignal(t*settings.dt))

        solution = sim.step(solution, actuation)

        # Log data
        minIdx = np.argmin(solution.reshape(-1,3)[:,2])
        log["minZ"].append(solution.reshape(-1,3)[minIdx])
        log["com"].append(solution.reshape(-1,3).mean(axis=0))
        # basePos.append(np.mean(solution.reshape(-1,3)[sim.baseIdx], axis=0))

        if visualize:
            if settings.meshType == "Tetrahedron":
                IO.save_tet_VTU(f"{settings.outputFolder}/{settings.meshFileName}_{t}", solution.reshape(-1,3), sim.elements, sim.visuals)
            elif settings.meshType == "Hexahedron":
                IO.save_hex_VTU(f"{settings.outputFolder}/{settings.meshFileName}_{t}", solution.reshape(-1,3), sim.elements, sim.visuals)

            sim.display(solution.reshape(-1,3), filename=f"{settings.outputFolder}/{settings.meshFileName}_{t:04d}.png", spp=4)
            print(f"Time until simulation step {t}: {(time.time()-begin):.2f}s")

    # Concatenate log data
    for k in log.keys():
        if isinstance(log[k], list):
            log[k] = np.stack(log[k], axis=0)

    # Loss computation based on maximum height after 0.5s
    if settings.numSteps*settings.dt > 0.5:
        log["maxPos"] = log["minZ"][int(0.5/(settings.dt)):].max()
    log["distance"] = log["com"][-1, 0] - log["com"][0, 0]
    
    # Store jump signal
    timeAxis = np.arange(settings.numSteps)*settings.dt
    np.savetxt(f"{settings.outputFolder}/plots/minPos.csv", np.concatenate([timeAxis.reshape(-1,1), log["minZ"]], axis=-1), delimiter=',')
    print(f"Average time per simulation step: {1e3*(time.time()-begin)/settings.numSteps:.1f}ms")

    if visualize:
        IO.save_pvd(settings.meshFileName, settings.outputFolder, settings.numSteps, settings.dt)
        export_mp4(f"{settings.outputFolder}", f"{settings.meshFileName}.mp4", int(1.0/settings.dt))

    # Plot trajectory
    timeAxis = np.arange(settings.numSteps)*settings.dt
    fig, axs = plt.subplots(1, 2, figsize=(110*mm, 40*mm))
    fig.subplots_adjust(wspace=0.3)
    # axs[0].plot(timeAxis, log["minZ"])
    axs[0].plot(timeAxis, log["com"][:, 2])
    axs[0].set_xlabel('Time (s)')
    axs[0].set_xlim([0, settings.numSteps*settings.dt])
    axs[0].set_ylabel('Z Position (m)')
    axs[0].grid()
    axs[0].ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    
    axs[1].plot(log["com"][:, 0], log["com"][:, 2])
    axs[1].set_xlabel('X Position (m)')
    axs[1].set_ylabel('Z Position (m)')
    axs[1].grid()
    axs[1].set_xlim([np.min(log["com"][:, 0]), np.max(log["com"][:, 0])])
    axs[1].ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    fig.savefig(f"{settings.outputFolder}/opt_results/traj_{log['maxPos']:.3f}m.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Plot actuation signal
    fig, ax = plt.subplots(figsize=(60*mm, 40*mm))
    timeAxis = np.arange(settings.numSteps)*settings.dt
    ax.plot(timeAxis, actSignal(timeAxis))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Actuation Signal (-)')
    ax.set_xlim([0, settings.numSteps*settings.dt])
    ax.grid()
    fig.savefig(f"{settings.outputFolder}/opt_results/act_{log['maxPos']:.3f}m.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    if visualize:
        export_mp4(f"{settings.outputFolder}", f"{settings.meshFileName}.mp4", int(1.0/settings.dt))
    
    return log


if __name__ == "__main__":
    settings.verbose_print()

    # Run unactuated simulation
    settings.outputFolder = "output/hopping_leg_unactuated"
    log = main(settings, stepTime=[0.0, 0.541], startVal=[0.0, 0.0], endVal=[0.0, 0.0], visualize=True)
    settings.outputFolder = "output/hopping_leg"
    log = main(settings, stepTime=[0.271, 0.541], startVal=[0.0, 0.69], endVal=[1.05, 0.0], visualize=True)
    
