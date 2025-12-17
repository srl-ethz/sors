################################################################################
# Jumping-leg loss-landscape search.
#
# Evaluates the hopping-leg rollout from `d19_hopping_leg.main()` over a parameter
# grid (and optionally Nelder–Mead) to maximize jump height. For each parameter
# vector x = [stepTime0, stepTime1, startVal1, endVal0], the script runs a full
# simulation, converts the objective to a scalar loss (negative max height), and
# logs values/losses to disk. After the sweep it visualizes 2D slices of the
# 4D loss landscape and renders the best-found policy.
################################################################################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from matplotlib import cm
matplotlib.use('agg')
import time, os, sys
sys.path.append('./cpp/build')
from py_sors import SimulationSettings
from d05_hopping_leg import main
# Matplotlib settings for consistent plotting
plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif'})#, "font.serif": ['Computer Modern']})
mm = 1 / 25.4

################################################################################
# Simulation settings
settings = SimulationSettings()
settings.description = "Optimization of a hopping leg that is muscle-actuated."
settings.meshFilePath = "hopping_leg.msh"
settings.outputFolder = "output/hopping_leg"
settings.meshFileName = settings.meshFilePath.split("/")[-1]
settings.meshType = "Hexahedron"
settings.dt = 2e-2
settings.numSteps = 50
settings.substeps = 25
settings.CFLtimeSteppingFlag = False
settings.timeSteppingScheme = "backward_euler" # "backward_euler", "crank_nicolson"
settings.solverMethod = "minimize_SQP"
settings.verbose = 0
settings.numThreads = 4
################################################################################

if __name__ == "__main__":
    settings.verbose_print()

    startTime = time.time()
    values, losses = [], []
    def f (x):
        maxPos = 0.0
        try:
            log = main(settings, stepTime=x[0:2], startVal=np.array([0, x[2]]), endVal=np.array([x[3], 0]), visualize=False)
            error = -log["maxPos"]
        except Exception as e:
            print(f"Error during simulation: {e}")
            maxPos = 0.0
            error = np.inf
        values.append(x)
        losses.append(error)

        print(f"\nTime {time.time()-startTime:.2f}s iteration {len(losses):03d}: \tParameters: [{x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f}, {x[3]:.4f}], Maximum Position: {maxPos:.6f}m")

        lowestLosses = losses.copy()
        for i in range(1, len(losses)):
            lowestLosses[i] = lowestLosses[i-1] if losses[i] >= lowestLosses[i-1] else losses[i]

        fig, ax = plt.subplots(figsize=(60*mm, 40*mm))
        ax.plot(losses, marker='o', linewidth=0, markersize=1, alpha=1, c='k', label="Loss")
        ax.plot(lowestLosses, c='r', linewidth=1, label="Lowest Loss")
        ax.grid()
        ax.set_xlabel("Iteration (-)")
        ax.set_ylabel("Loss (-)")
        ax.set_xlim(0, len(losses))
        ax.ticklabel_format(style="sci", axis='both', scilimits=(0,0))
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=2)

        fig.savefig(f"{settings.outputFolder}/opt_results/loss.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        return error

    ### Run Nelder Mead optimization
    # res = scipy.optimize.minimize(
    #     f, 
    #     [0.5, -0.1, 0.2], 
    #     bounds=[(0.1, 1.4), (-1, 1), (-1, 1)], 
    #     method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-9}
    # )
    # print(f"Optimization result: {res.x}, Loss: {res.fun:.4e}m")

    ### Grid Search
    N = 7
    stepTimes0 = np.linspace(0.25, 0.75, N)
    stepTimes1 = np.linspace(0.25, 0.75, N)
    startVals = np.linspace(0.0, 1.5, N)
    endVals = np.linspace(0.0, 1.5, N)

    lowestLoss = np.inf
    for stepTime0 in stepTimes0:
        for stepTime1 in stepTimes1:
            for startVal in startVals:
                for endVal in endVals:
                    f([stepTime0, stepTime1, startVal, endVal])
                    if losses[-1] < lowestLoss:
                        lowestLoss = losses[-1]
                        print(f"New best parameters: {[stepTime0, stepTime1, startVal, endVal]} with max height {-lowestLoss:.4f}m")

            np.savetxt(f"{settings.outputFolder}/opt_results/values.txt", np.array(values), delimiter=',')
            np.savetxt(f"{settings.outputFolder}/opt_results/losses.txt", np.array(losses), delimiter=',')

    values = np.loadtxt(f"{settings.outputFolder}/opt_results/values.txt", delimiter=',')
    losses = np.loadtxt(f"{settings.outputFolder}/opt_results/losses.txt", delimiter=',')

    resX = values[np.argmin(losses)]
    print(f"Best parameters found: {resX} with max height {-np.min(losses):.4f}m")

    ### Prepare Grids and Losses
    st0Grid = np.array(values)[:,0].reshape(N,N,N,N)
    st1Grid = np.array(values)[:,1].reshape(N,N,N,N)
    svGrid = np.array(values)[:,2].reshape(N,N,N,N)
    evGrid = np.array(values)[:,3].reshape(N,N,N,N)
    losses = np.array(losses).reshape(N,N,N,N)
    st0Index = losses.argmin()//(N*N*N)
    st1Index = (losses.argmin()//(N*N))%N
    svIndex = (losses.argmin()//N)%N
    evIndex = losses.argmin()%N

    ### Plot 2D Loss Landscapes
    nlevel = 15
    fig, axs = plt.subplots(1, 4, figsize=(4*66*mm, 40*mm))
    plt.subplots_adjust(wspace=0.8)

    cax = axs[0].contourf(st0Grid[:,st1Index,:,evIndex].transpose(), svGrid[:,st1Index,:,evIndex].transpose(), losses[:,st1Index,:,evIndex].transpose(), levels=nlevel, cmap=cm.coolwarm)
    axs[0].set_xlabel("Steptime 0 (s)")
    axs[0].set_ylabel("Start Value (-)")
    fig.colorbar(cax)

    cax = axs[1].contourf(st0Grid[:, st1Index, svIndex].transpose(), evGrid[:, st1Index, svIndex].transpose(), losses[:, st1Index, svIndex].transpose(), levels=nlevel, cmap=cm.coolwarm)
    axs[1].set_xlabel("Steptime 0 (s)")
    axs[1].set_ylabel("End Value (-)")
    fig.colorbar(cax)

    cax = axs[2].contourf(svGrid[st0Index, st1Index].transpose(), evGrid[st0Index, st1Index].transpose(), losses[st0Index, st1Index].transpose(), levels=nlevel, cmap=cm.coolwarm)
    axs[2].set_xlabel("Start Value (-)")
    axs[2].set_ylabel("End Value (-)")
    fig.colorbar(cax)

    cax = axs[3].contourf(st0Grid[:, :, svIndex, evIndex].transpose(), st1Grid[:, :, svIndex, evIndex].transpose(), losses[:, :, svIndex, evIndex].transpose(), levels=nlevel, cmap=cm.coolwarm)
    axs[3].set_xlabel("Steptime 0 (s)")
    axs[3].set_ylabel("Steptime 1 (s)")
    fig.colorbar(cax)

    fig.savefig(f"{settings.outputFolder}/opt_results/loss_landscape.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    main(settings, resX[0:2],  np.array([0, resX[2]]), np.array([resX[3], 0]), visualize=True)
