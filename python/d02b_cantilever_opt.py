################################################################################
# Cantilever SysID sweep runner.
#
# Launches/continues a Weights & Biases sweep for cantilever sim-to-real system
# identification by sampling material/damping parameters, running the evaluation
# in d05a_cantilever_sim2real.py, and logging error metrics + diagnostic plots.
################################################################################

import sys, traceback, argparse, yaml
from datetime import datetime
import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
sys.path.append('./cpp/build')
from d02a_cantilever_sim2real import settings
from d02a_cantilever_sim2real import main as run_simulation
# Matplotlib settings for consistent plotting
plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
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
    "alpha":            50.0,
}
################################################################################

def run_sweep ():
    """Execute one W&B sweep trial and log sim-to-real results.

    Initializes a W&B run, pulls the current hyperparameters (density, stiffness,
    Poisson's ratio, damping), runs the cantilever sim-to-real evaluation, and
    uploads summary statistics and trajectory/overlay plots to W&B.
    """
    try:
        wandb.init(config=configs)

        ### Run simulation
        density = wandb.config.get("density")
        youngsModulus = wandb.config.get("youngsModulus")
        poissonsRatio = wandb.config.get("poissonsRatio")
        alpha = wandb.config.get("alpha")   

        log = run_simulation(settings, density=density, youngsModulus=youngsModulus, poissonsRatio=poissonsRatio, alpha=alpha, visualize=False)

        # Plot results in wandb
        figFullTraj = []
        for i, sp in enumerate(log["simPosFull"]):
            fig, ax = plt.subplots(figsize=(3,2))
            ax.plot(sp[:,2])
            ax.grid()
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Z Position (m)')
            ax.set_title(f"Weight {log["weights"][i]:.3f}kg")
            figFullTraj.append(wandb.Image(fig))

        figSim2Real = []
        for i, (sim, real) in enumerate(zip(log["simPos"], log["realPos"])):
            fig, axs = plt.subplots(1, 3, figsize=(200*mm,40*mm))
            fig.subplots_adjust(wspace=0.5)
            axs[0].plot(sim[:, :, 0].mean(-1), label='Simulated X')
            axs[0].plot(real[:settings.numSteps, :, 0].mean(-1), label='Real X', linestyle='--')
            axs[0].set_xlabel('Timestep')
            axs[0].set_ylabel('X Position (m)')

            axs[1].plot(sim[:, :, 2].mean(-1), label='Simulated Z')
            axs[1].plot(real[:settings.numSteps, :, 2].mean(-1), label='Real Z', linestyle='--')
            axs[1].set_xlabel('Timestep')
            axs[1].set_ylabel('Z Position (m)')

            axs[2].plot(log["errors"][i].mean(-1), label='Error')
            axs[2].set_xlabel('Timestep')
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

            axs[1].set_title(f"Simulation vs Real Data for Weight {log["weights"][i]:.3f}kg")
            figSim2Real.append(wandb.Image(fig))

        wandb.log({
            "error_mean": np.mean(log["errors"].flatten()),
            "error_std": np.std(log["errors"].flatten()),
            "sim2real": figSim2Real,
            "fullTraj": figFullTraj,
        })
        plt.close('all')

    except:
        # Exit gracefully, so wandb logs the problem
        print(traceback.print_exc())
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Optimization for Cantilever.")
    parser.add_argument('--sweepId', type=str, default=None, help='Sweep ID to continue running previous WandB sweep')
    parser.add_argument('--sweepConfig', type=str, default="python/configs/config_cantilever.yml", help='Config file for hyperparameter sweep')
    parser.add_argument('--wandbEntity', type=str, default="", help='WandB sweep entity.')
    args = parser.parse_args()

    with open(args.sweepConfig, 'r', encoding='utf-8') as stream:
        try:
            sweepConfig = yaml.safe_load(stream)
            print(sweepConfig)
        except yaml.YAMLError as exc:
            print(exc)

    # Name the sweep based on current date
    sweepConfig['name'] = f"sweep_cantilever_{datetime.now().strftime('%Y%m%d')}"
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





