################################################################################
# SoPrA SysID hyperparameter sweep runner.
#
# Launches/continues a W&B sweep defined in a YAML config (config_sopra.yaml), 
# runs the quasistatic sim-to-real evaluation for each sampled parameter set, 
# and logs error metrics plus diagnostic plots for sweep optimization. The 
# simulation environment is defined in d04_sopra.py.
################################################################################

import sys, traceback, argparse, yaml
from datetime import datetime

import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

sys.path.append('./cpp/build')
from d04a_sopra_quasistatic import settings
from d04a_sopra_quasistatic import main as run_simulation

# Matplotlib settings for consistent plotting
plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif'}) #, "font.serif": ['Computer Modern']})
mm = 1 / 25.4

################################################################################
# Sweep config
configs = {
    "density":          1070,
    "youngsModulus":    250e3,
    "poissonsRatio":    0.49,
    "alpha":            75.0,
}
################################################################################

def run_sweep ():
    """Execute a single W&B sweep trial for SoPrA system identification.

    Initializes a W&B run, pulls the current hyperparameters from `wandb.config`,
    runs the quasistatic sim-to-real matching (retrying with increased substeps on
    failure), and logs scalar metrics and plots back to W&B.
    """
    try:
        wandb.init(config=configs)

        ### Run simulation
        density = wandb.config.get("density")
        youngsModulus = wandb.config.get("youngsModulus")
        poissonsRatio = wandb.config.get("poissonsRatio")

        substeps = 10
        for _ in range(3): # Try at most 5 times
            try: 
                print(f"Running with substeps {substeps}")
                log = run_simulation(settings, density=density, youngsModulus=youngsModulus, poissonsRatio=poissonsRatio, alpha=75.0, substeps=substeps, visualize=False)
                break
            except:
                # Try again with more substeps
                substeps = int(substeps * 2)
                print(f"### Exception during simulation, trying again with more substeps {substeps}")
                log = {"tipPos": [], "simPos": [], "realPos": [], "errors": [1e3]}

        # Plot results in wandb
        fig0, ax = plt.subplots(figsize=(60*mm, 60*mm))
        s = 20
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
        for i, (sim, real) in enumerate(zip(log["simPos"], log["realPos"])):
            for j, (sp, rp) in enumerate(zip(sim, real)):
                ax.scatter(rp[0], rp[1], marker='o', s=s, c=colors[i%6], alpha=(j+1)/log["realPos"].shape[1], zorder=2)
                ax.scatter(sp[0], sp[1], marker='x', s=s, c=colors[i%6], alpha=(j+1)/log["simPos"].shape[1], zorder=2)
        ax.scatter([], [], marker='o', s=s, c='k', label="Real", zorder=2)
        ax.scatter([], [], marker='x', s=s, c='k', label="Simulated", zorder=2)
        ax.grid()
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_xlim(-0.08, 0.06)
        ax.set_ylim(-0.07, 0.07)
        ax.ticklabel_format(style="sci", axis='both', scilimits=(0,0))
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=2)

        tipFigs = []
        for traj in log["tipPos"]:
            timeAxis = np.arange(settings.numSteps) * settings.dt
            fig, axs = plt.subplots(1, 2, figsize=(120*mm, 40*mm))
            fig.subplots_adjust(wspace=0.5)
            axs[0].plot(traj[:, 0], linestyle='-')
            axs[0].set_xlabel('Timestep')
            axs[0].set_ylabel('X Position (m)')

            axs[1].plot(traj[:, 1], linestyle='-')
            axs[1].set_xlabel('Timestep')
            axs[1].set_ylabel('Y Position (m)')

            axs[0].grid()
            axs[1].grid()
            axs[0].ticklabel_format(axis='both', style='sci', scilimits=(0,0))
            axs[1].ticklabel_format(axis='both', style='sci', scilimits=(0,0))
            tipFigs.append(wandb.Image(fig))

        wandb.log({
            "substeps": substeps,
            "error_mean": np.mean(log["errors"]),
            "error_std": np.std(log["errors"]),
            "sim2real": wandb.Image(fig0),
            "tipTrajectory": tipFigs,
        })
        plt.close('all')

    except:
        # Exit gracefully, so wandb logs the problem
        print(traceback.print_exc())
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Optimization for SoPrA.")
    parser.add_argument('--sweepId', type=str, default=None, help='Sweep ID to continue running previous WandB sweep')
    parser.add_argument('--sweepConfig', type=str, default="python/configs/config_sopra.yml", help='Config file for hyperparameter sweep')
    parser.add_argument('--wandbEntity', type=str, default="", help='WandB sweep entity.')
    args = parser.parse_args()

    with open(args.sweepConfig, 'r', encoding='utf-8') as stream:
        try:
            sweepConfig = yaml.safe_load(stream)
            print(sweepConfig)
        except yaml.YAMLError as exc:
            print(exc)

    # Name the sweep based on current date
    sweepConfig['name'] = f"sweep_sopra_{datetime.now().strftime('%Y%m%d')}"
    settings.verbose_print()

    # Continue running existing sweeps
    if args.sweepId is not None:
        sweepId = args.sweepId
    else:
        sweepId = wandb.sweep(sweepConfig, entity=args.wandbEntity)
    # Multiple machines can run the same sweep
    print(f"### Sweep ID: {sweepId}")
    ### The configs from wandb are updated with sweep config!
    wandb.agent(sweepId, function=run_sweep)





