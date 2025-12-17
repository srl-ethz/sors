################################################################################
# Cantilever sim-to-real matching with experimental data.
#
# Replays measured tip deflections under different payloads, aligns the marker
# frames to the simulated beam after a gravity-settling phase, and evaluates
# marker-wise L2 tracking error over time (with optional VTU export + rendering).
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
from py_sors import Params, IO, SimulationSettings
from d02_cantilever import BeamEnv
from _renderer import export_mp4
from _utils import best_fit_transform, compute_marker_interpolation, marker_interpolation, bcolors
plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif'}) #, "font.serif": ['Computer Modern']})
mm = 1 / 25.4

################################################################################
# Simulation Settings
settings = SimulationSettings()
settings.description = "Demo of passive soft cantilever."
settings.meshFilePath = "cantilever.msh"
settings.outputFolder = "output/cantilever"
settings.meshFileName = settings.meshFilePath.split("/")[-1]
settings.meshType = "Tetrahedron"
settings.dt = 1e-2
settings.numSteps = 150
settings.substeps = 5
settings.CFLtimeSteppingFlag = False
settings.timeSteppingScheme = "crank_nicolson" # "backward_euler", "crank_nicolson"
settings.solverMethod = "minimize_newton"
settings.verbose = 0
settings.numThreads = 8
# Dataset Settings
dataPath = "data/cantilever/qs_real.npy" # Path to real data
################################################################################

def main (settings, density=1070.0, youngsModulus=263824.0, poissonsRatio=0.49, alpha=20, visualize=True):
    """Run cantilever sim-to-real evaluation against a recorded marker dataset.

    The pipeline (i) settles the beam under gravity using a damped integrator,
    (ii) aligns real markers to the simulated frame via best-fit transforms,
    (iii) applies the measured payload as distributed tip forces, then (iv)
    simulates unloading dynamics and computes per-marker tracking errors.

    Returns:
        dict: Log dictionary with:
            - "errors" (np.ndarray): Per-marker L2 errors, shape [numTraj, T, nMarkers].
            - "simPos" (np.ndarray): Simulated marker positions, shape [numTraj, T, nMarkers, 3].
            - "realPos" (np.ndarray): Aligned real marker positions, shape [numTraj, T, nMarkers, 3].
            - "simPosFull" (np.ndarray): Mean marker position trajectory (debug), shape [numTraj, T_full, 3].
            - "weights" (np.ndarray): Payload mass per trajectory (kg), shape [numTraj].
    """
    if not os.path.exists(f"{settings.outputFolder}/plots"):
        os.makedirs(f"{settings.outputFolder}/plots")

    # Set up simulation environment, one energy conserving and one dampened.
    settings.timeSteppingScheme = "crank_nicolson"
    sim = BeamEnv(settings, density=density, youngsModulus=youngsModulus, poissonsRatio=poissonsRatio, alpha=alpha)
    settings.timeSteppingScheme = "backward_euler"
    simDampened = BeamEnv(settings, density=density, youngsModulus=youngsModulus, poissonsRatio=poissonsRatio, alpha=50)
    actuation = Params(["vertexForce"], [np.zeros(sim.initialVertices.shape)])

    # Load Real Data
    numTraj = 17
    loadedData = np.load(dataPath, allow_pickle=True)[()]
    realMarkers = loadedData['data'][:numTraj]
    realWeight = loadedData['weight'][:numTraj]
    print(f"{bcolors.HEADER}Loaded {realMarkers.shape[0]} trajectories of length {realMarkers.shape[1]}.{bcolors.ENDC}")

    # Simulated marker locations
    measuredMarkers = np.array([
        [0.0247, 0.0, 0.015],
        [0.0447, 0.0, 0.015],
        [0.0648, 0.0, 0.015],
        [0.0447, 0.015, 0.03],
        [0.0648, 0.006, 0.03],
        [0.0648, 0.024, 0.03],
        [0.0247, 0.03, 0.006],
        [0.0247, 0.03, 0.024],
        [0.0447, 0.03, 0.006],
        [0.0648, 0.03, 0.015],
    ])

    closestElements, interpolationCoeffs = compute_marker_interpolation(sim.initialVertices, sim.elements, measuredMarkers)
    initialSimMarkers = marker_interpolation(sim.initialVertices, sim.elements, closestElements, interpolationCoeffs)

    # Apply gravity
    solution = sim.initialVertices.copy().flatten()
    gravitySteps = 25
    for t in range(gravitySteps):
        solution = simDampened.step(solution, dt=1e-2)

    initialSimMarkers = marker_interpolation(solution.reshape(-1,3), sim.elements, closestElements, interpolationCoeffs)

    # Align real markers to simulated frame with average deflection under gravity.
    Rs = []
    ts = []
    for i in range(numTraj):
        # Align the real markers to the simulated markers using best fit transform
        T, R, t = best_fit_transform(realMarkers[i,-1], initialSimMarkers)
        Rs.append(R)
        ts.append(t)
    R = np.mean(Rs, axis=0)
    t = np.mean(ts, axis=0)
    realAlignedMarkers = np.einsum('ij, tbkj -> tbki', R, realMarkers) + t.reshape(1, 1, 1, -1)

    ### RUN SIMULATION ###
    # Initial state
    initialSolution = solution.copy()
    simulatedMarkers = []
    simPosFull = []
    for i in range(len(realMarkers)):
        solution = initialSolution.copy()
        # Store VTU
        IO.save_tet_VTU(f"{settings.outputFolder}/{settings.meshFileName}_{i:02d}_{0:04d}", solution.reshape(-1,3), sim.elements)
        if visualize:
            sim.display(solution.reshape(-1,3), filename=f"{settings.outputFolder}/{settings.meshFileName}_{i:02d}_{0:04d}.png", spp=4)

        begin = time.time()
        sp = [marker_interpolation(solution.reshape(-1,3), sim.elements, closestElements, interpolationCoeffs).mean(0)]
        # Apply weight
        simDampened.systemEnergy.set_initial_deformation(solution.reshape(-1,3))
        loadingSteps = 50
        for t in range(loadingSteps):
            # Actuation
            vertexForce = np.zeros(sim.initialVertices.shape)
            vertexForce[sim.tipIdx, 2] = min(10*t, loadingSteps)/loadingSteps * -realWeight[i]*9.81 / len(sim.tipIdx)  # Distribute weight evenly across tip vertices
            actuation.set_param("vertexForce", vertexForce)

            solution = simDampened.step(solution, actuation, dt=1e-2)
            sp.append(marker_interpolation(solution.reshape(-1,3), sim.elements, closestElements, interpolationCoeffs).mean(0))
        
        # Unload
        sim.systemEnergy.set_initial_deformation(solution.reshape(-1,3)) # Set the previous position
        markerPos = [marker_interpolation(solution.reshape(-1,3), sim.elements, closestElements, interpolationCoeffs)]
        for t in range(1, settings.numSteps):
            solution = sim.step(solution)

            simMarkers = marker_interpolation(solution.reshape(-1,3), sim.elements, closestElements, interpolationCoeffs)
            markerPos.append(simMarkers)
            sp.append(marker_interpolation(solution.reshape(-1,3), sim.elements, closestElements, interpolationCoeffs).mean(0))

            # Store VTU
            IO.save_tet_VTU(f"{settings.outputFolder}/{settings.meshFileName}_{i:02d}_{t:04d}", solution.reshape(-1,3), sim.elements)
            if visualize:
                print(f"Time until simulation step {t}: {(time.time()-begin):.2f}s")
                sim.display(solution.reshape(-1,3), filename=f"{settings.outputFolder}/{settings.meshFileName}_{i:02d}_{t:04d}.png", spp=4)
                
        markerPos = np.stack(markerPos, axis=0)
        simulatedMarkers.append(markerPos)
        sp = np.stack(sp, axis=0)
        simPosFull.append(sp)
        print(f"Average time per simulation step: {1e3*(time.time()-begin)/settings.numSteps:.1f}ms with total time {time.time()-begin:.2f}s")

        # Visualize marker positions between simulation and reality
        timeAxis = np.arange(len(markerPos)) * settings.dt
        fig, axs = plt.subplots(1, 3, figsize=(200*mm,40*mm))
        fig.subplots_adjust(wspace=0.5)
        axs[0].plot(timeAxis, markerPos[:, :, 0].mean(-1), label='Simulated X')
        axs[0].plot(timeAxis, realAlignedMarkers[i, :settings.numSteps, :, 0].mean(-1), label='Real X', linestyle='--')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('X Position (m)')
        
        axs[1].plot(timeAxis, markerPos[:, :, 2].mean(-1), label='Simulated Z')
        axs[1].plot(timeAxis, realAlignedMarkers[i, :settings.numSteps, :, 2].mean(-1), label='Real Z', linestyle='--')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Z Position (m)')
        
        axs[2].plot(timeAxis, np.linalg.norm(markerPos - realAlignedMarkers[i,:settings.numSteps], axis=-1).mean(-1), label='Error')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('L2 Error (m)')
        
        axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2)
        axs[1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2)
        axs[2].legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2)
        axs[0].grid()
        axs[1].grid()
        axs[2].grid()
        axs[0].ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        axs[1].ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        axs[2].ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        axs[1].set_title(f"Simulation vs Real Data for Weight {realWeight[i]:.3f}kg")
        fig.savefig(f"{settings.outputFolder}/plots/sim2real_{realWeight[i]:.3f}kg.png", dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(3,2))
        ax.plot(sp[:,2])
        ax.grid()
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Z Position (m)')
        ax.set_title(f"Weight {realWeight[i]:.3f}kg")
        fig.savefig(f"{settings.outputFolder}/plots/full_oscillation_{realWeight[i]:.3f}kg.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Store PVD for VTU files
        IO.save_pvd(settings.meshFileName, settings.outputFolder, settings.numSteps, settings.dt)
        if visualize:
            export_mp4(f"{settings.outputFolder}", f"{settings.meshFileName}.mp4", int(1.0/settings.dt))

    simulatedMarkers = np.stack(simulatedMarkers, axis=0)
    simPosFull = np.stack(simPosFull, axis=0)

    # Calculate error based on L2
    errors = np.linalg.norm(simulatedMarkers - realAlignedMarkers[:,:settings.numSteps], axis=-1)

    # Store CSV Data
    csvSimMarkers, csvRealMarkers, csvErrors = [], [], []
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            csvSimMarkers.append(
                np.append([i, j*settings.dt], simulatedMarkers[i,j].flatten())
            )
            csvRealMarkers.append(
                np.append([i, j*settings.dt], realAlignedMarkers[i,j].flatten())
            )
            csvErrors.append(
                np.append([i, j*settings.dt], errors[i,j])
            )
    csvSimMarkers = np.stack(csvSimMarkers, axis=0)
    csvRealMarkers = np.stack(csvRealMarkers, axis=0)
    csvErrors = np.stack(csvErrors, axis=0)
    header = "Trajectory Idx, Time (s)"
    header3d = "Trajectory Idx, Time (s)"
    for i in range(errors.shape[2]):
        header += f", marker {i}"
        header3d += f", x{i}, y{i}, z{i}"
    np.savetxt(f"{settings.outputFolder}/plots/simMarker.csv", csvSimMarkers, delimiter=',', header=header3d)
    np.savetxt(f"{settings.outputFolder}/plots/realMarker.csv", csvRealMarkers, delimiter=',', header=header3d)
    np.savetxt(f"{settings.outputFolder}/plots/errors.csv", csvErrors, delimiter=',', header=header)

    log = {
        "errors": errors, # Shape [numTraj, T, nMarkers]
        "simPos": simulatedMarkers, # Shape [numTraj, T, nMarkers, 3]
        "realPos": realAlignedMarkers[:,:settings.numSteps],
        "simPosFull": simPosFull,  # Shape [numTraj, T, 3], average tip marker movement, for debugging
        "weights": realWeight,
    }

    return log


if __name__ == "__main__":
    settings.verbose_print()

    log = main(settings, density=1089.8, youngsModulus=236638.0, poissonsRatio=0.316, alpha=6.423, visualize=True)
    print(f"Trajectory Errors: {np.mean(log["errors"]):.4e}m +- {np.std(log["errors"]):.4e}m")
