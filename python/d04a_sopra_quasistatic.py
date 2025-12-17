################################################################################
# SoPrA quasistatic sim-to-real matching.
#
# Replays measured pressure trajectories on the simulated SoPrA arm and compares
# simulated tip positions against real-world measurements.
################################################################################

import sys
import os
import time
import scipy
import pandas as pd
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('./cpp/build')
from py_sors import Params, IO
from _renderer import export_mp4
from d04_sopra import SopraEnv, settings
plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif'}) #, "font.serif": ['Computer Modern']})
mm = 1 / 25.4

################################################################################
# Dataset Settings
datatype = "singleChamber" # SingleChamber, random
numTraj = 10        # Number of trajectories to use
numIdx = 6          # Number of points in each trajectory to use
################################################################################

def main (settings, density=1070.0, youngsModulus=263824.0, poissonsRatio=0.49, alpha=50.0, substeps=1, visualize=False):
    """Run quasistatic sim-to-real evaluation for the SoPrA arm.

    Loads real pressure–tip trajectories, applies quasistatic pressure ramps in
    simulation, and evaluates the positional error between simulated and measured
    tip positions.

    Returns:
        dict:
            Dictionary containing:
            - "errors": XY tip position errors (m),
            - "simPos": simulated quasistatic tip positions (m),
            - "realPos": measured tip positions (m),
            - "tipPos": simulated tip trajectories during pressure ramps (m).
    """
    settings.substeps = substeps
    # Load Real Data
    data = {"xe_BT": [], "ye_BT": [], "ze_BT": [], "pressure_0": [], "pressure_1": [], "pressure_2": [], "pressure_3": [], "pressure_4": [], "pressure_5": []}
    if datatype == "singleChamber":
        for i in range(6): # Six chambers to actuate separately
            df = pd.read_csv(f"data/sopra_quasistatic/20211025194737-sopra-test-sawtooth-v{i}-700.csv", header=0)
            d = df.to_dict(orient='list')
            d["idx"] = d.pop("Unnamed: 0") # First column is index
            
            for key in data.keys():
                data[key].append(d[key])

    elif datatype == "random":
        for i in [2,3,5,6,7,9,10,11,12,13]: # Stable trajectories for our simulation
            df = pd.read_csv(f"data/sopra_quasistatic/20211025205642-sopra-test-sawtooth50-vRand-ii{i*100}-650.csv", header=0)
            d = df.to_dict(orient='list')
            d["idx"] = d.pop("Unnamed: 0") # First column is index
            
            for key in data.keys():
                data[key].append(d[key])
    else:
        raise ValueError(f"Unknown datatype {datatype}")

    # Position of base to tip vector (BT)
    realPos = np.stack([data["xe_BT"], data["ye_BT"], data["ze_BT"]], axis=-1)[:numTraj, :numIdx] / 1e3 # Convert mm to m
    realP = np.stack([data[f"pressure_{i}"] for i in range(6)], axis=-1)[:numTraj, :numIdx] * 1e2 # Convert mbar to Pa
    chamberMapping = [1, 4, 3, 5, 0, 2] # Fixed mapping from sim to real chambers

    if not os.path.exists(f"{settings.outputFolder}/plots"):
        os.makedirs(f"{settings.outputFolder}/plots")

    # DEBUG
    idx = 5
    realPos = realPos[:,idx:idx+1]
    realP = realP[:,idx:idx+1]

    # Store realPos in csv
    csvRealPos = []
    for i in range(realPos.shape[0]):
        for j in range(realPos.shape[1]):
            csvRealPos.append(np.array([i, realPos[i,j,0], realPos[i,j,1], realPos[i,j,2]]))
    csvRealPos = np.stack(csvRealPos, axis=0)
    np.savetxt(f"{settings.outputFolder}/plots/realPos.csv", csvRealPos)
    
    # Run simulation
    simPos = [] # Final quasistatic tip solutions of all trajectories [N, M, 3]
    tipPos = []
    for i in range(realPos.shape[0]):
        sim = SopraEnv(settings, density, youngsModulus, poissonsRatio, alpha)
        actuation = Params(["pressure"], [np.zeros([6,1])]) # 6 chambers

        # Initial state
        solution = sim.initialVertices.copy().flatten()
        if visualize:
            IO.save_tet_VTU(f"{settings.outputFolder}/{settings.meshFileName}_{i:02d}_0", solution.reshape(-1,3), sim.elements)
            sim.display(solution.reshape(-1,3), markers=realPos[i], filename=f"{settings.outputFolder}/{settings.meshFileName}_{i:02d}_{0:04d}.png", spp=4)

        # Run simulation
        sm = []
        tp = []
        timestep = 1
        prevP = np.zeros(6)
        for j in range(realPos.shape[1]):
            begin = time.time()
            for t in range(1, settings.numSteps):
                # Actuation
                actuation.set_param(
                    "pressure", 
                    min(1, settings.numSteps*settings.dt/0.8 * t/settings.numSteps) * (realP[i, j][chamberMapping]-prevP) + prevP
                ) # 0.8s ramp

                solution = sim.step(solution, actuation)
                tp.append(solution.reshape(-1,3)[sim.tipIdx].mean(axis=0))

                if visualize:
                    IO.save_tet_VTU(f"{settings.outputFolder}/{settings.meshFileName}_{i:02d}_{timestep}", solution.reshape(-1,3), sim.elements)
                    sim.display(solution.reshape(-1,3), markers=realPos[i], filename=f"{settings.outputFolder}/{settings.meshFileName}_{i:02d}_{timestep:04d}.png", spp=4)
                    print(f"Time until step {timestep}: {(time.time()-begin):.2f}s, Pressures: {actuation.get_value('pressure').flatten()}")

                timestep += 1

            sm.append(np.mean(solution.reshape(-1,3)[sim.tipIdx], axis=0))
            prevP = realP[i, j][chamberMapping]
            print(f"Simulation Trajectory {i} Point {j}: Average time per step: {1e3*(time.time()-begin)/settings.numSteps:.1f}ms \t- Total Time {time.time()-begin:.2f}s")
        
        tipPos.append(tp)
        simPos.append(sm)
    tipPos = np.stack(tipPos, axis=0) # [N, T, 3]
    simPos = np.stack(simPos, axis=0) # [N, M, 3]
    errors = np.linalg.norm(simPos[...,:2] - realPos[...,:2], axis=-1) # Only consider x and y position for error.

    # Visualize simulated and real markers
    fig, ax = plt.subplots(figsize=(60*mm, 60*mm))
    s = 20
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    for i, (sim, real) in enumerate(zip(simPos, realPos)):
        for j, (sp, rp) in enumerate(zip(sim, real)):
            ax.scatter(rp[0], rp[1], marker='o', s=s, c=colors[i%6], alpha=(j+1)/realPos.shape[1], zorder=2)
            ax.scatter(sp[0], sp[1], marker='x', s=s, c=colors[i%6], alpha=(j+1)/simPos.shape[1], zorder=2)
    ax.scatter([], [], marker='o', s=s, c='k', label="Real", zorder=2)
    ax.scatter([], [], marker='x', s=s, c='k', label="Simulated", zorder=2)
    ax.grid()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(-0.08, 0.06)
    ax.set_ylim(-0.07, 0.07)
    ax.ticklabel_format(style="sci", axis='both', scilimits=(0,0))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=2)
    fig.savefig(f"{settings.outputFolder}/plots/sim_vs_real_xy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if visualize:
        export_mp4(f"{settings.outputFolder}", f"{settings.meshFileName}.mp4", int(1.0/settings.dt))

    # Store simPos in csv
    csvSimPos = []
    for i in range(simPos.shape[0]):
        for j in range(simPos.shape[1]):
            csvSimPos.append(np.array([i, simPos[i,j,0], simPos[i,j,1], simPos[i,j,2]]))
    csvSimPos = np.stack(csvSimPos, axis=0)
    np.savetxt(f"{settings.outputFolder}/plots/simPos.csv", csvSimPos)

    # Store the errors as csv.
    np.savetxt(f"{settings.outputFolder}/plots/errors.csv", errors)

    log = {
        "errors": errors,
        "simPos": simPos,
        "realPos": realPos,
        "tipPos": tipPos,
    }
    return log


if __name__ == "__main__":
    settings.verbose_print()
    
    log = main(settings, density=1167, youngsModulus=346.5e3, poissonsRatio=0.358, alpha=75, visualize=True)
    print(f"Errors Mean: {np.mean(log['errors']):.4e}m, Std: {np.std(log['errors']):.4e}m")

    
