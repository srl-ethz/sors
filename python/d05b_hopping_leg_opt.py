################################################################################
# Run hyperparameter optimization for the jumping leg experiment. 
# 
# Configurations should be defined in d19_hopping_leg.py, with hyperparameters 
# to optimize being defined in the config_sweep_leg.yaml file. The simulation 
# with proper metrics is defined in d19a_hopping_leg.py.
#
# For each W&B run, the script:
#   1) Reads actuation parameters (step times + start/end activations per muscle),
#   2) Executes a full simulation rollout without visualization,
#   3) Logs scalar metrics (max height, horizontal distance) and diagnostic plots
#      (height trajectory, actuation signal) back to W&B.
################################################################################

import sys, traceback, argparse, yaml
from datetime import datetime
import wandb
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('./cpp/build')
from d05_hopping_leg import settings
from d05_hopping_leg import main as run_simulation
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
    "stepTime0":    0.5,        # Time (s) at which muscle 0 transitions from startVal to endVal
    "startVal0":    0.0,        # Starting value for muscle 0
    "endVal0":      1.0,        # End value for muscle 0

    "stepTime1":    0.5,        # Time (s) at which muscle 1 transitions from startVal to endVal
    "startVal1":    1.0,        # Starting value for muscle 1
    "endVal1":      0.0,        # End value for muscle 1
}
################################################################################

def run_sweep ():
    """Run a single W&B sweep iteration (one parameter sample → one simulation).

    This function is passed to `wandb.agent(...)`. It initializes a W&B run using
    the default `configs` dict (overridden by the sweep controller), constructs
    the two-muscle actuation schedule expected by `run_simulation`, executes the
    rollout, and logs metrics/plots.

    Logged outputs (from `log` returned by `run_simulation`):
        - maxHeight:  log["maxPos"]      (float)
        - distance:   log["distance"]    (float)
        - heightTrajectory: plot of log["minZ"] vs time
        - actuationSignal: plot of log["actSignal"] vs time

    Notes:
        - Any exception is caught so the W&B run can be marked as failed while
          still capturing the traceback in logs.
        - This function assumes `settings.numSteps` and `settings.dt` are already
          configured in the imported `settings` object.
    """
    try:
        wandb.init(config=configs)

        ### Run simulation
        stepTime = [wandb.config.get("stepTime0"), wandb.config.get("stepTime1")]
        startVal = np.array([wandb.config.get("startVal0"), wandb.config.get("startVal1")])
        endVal = np.array([wandb.config.get("endVal0"), wandb.config.get("endVal1")])

        log = run_simulation(settings, stepTime, startVal, endVal, visualize=False)
        timeAxis = np.arange(settings.numSteps)*settings.dt

        # Plot results in wandb
        fig1, ax1 = plt.subplots()
        ax1.plot(timeAxis, log["minZ"])
        ax1.grid()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Height (m)")

        fig2, ax2 = plt.subplots()
        ax2.plot(timeAxis, log["actSignal"])
        ax2.grid()
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Actuation")

        wandb.log({
            "maxHeight": log["maxPos"],
            "distance": log["distance"],
            "heightTrajectory": wandb.Image(fig1),
            "actuationSignal": wandb.Image(fig2),
        })
        plt.close('all')

    except:
        # Exit gracefully, so wandb logs the problem
        print(traceback.print_exc())
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Optimization for jumping leg.")
    parser.add_argument('--sweepId', type=str, default=None, help='Sweep ID to continue running previous WandB sweep')
    parser.add_argument('--sweepConfig', type=str, default="python/configs/config_leg.yml", help='Config file for hyperparameter sweep')
    parser.add_argument('--wandbEntity', type=str, default="", help='WandB sweep entity.')
    args = parser.parse_args()

    with open(args.sweepConfig, 'r', encoding='utf-8') as stream:
        try:
            sweepConfig = yaml.safe_load(stream)
            print(sweepConfig)
        except yaml.YAMLError as exc:
            print(exc)

    # Name the sweep based on current date
    sweepConfig['name'] = f"sweep_leg_{datetime.now().strftime('%Y%m%d')}"
    settings.verbose_print()

    # Continue running existing sweeps
    if args.sweepId is not None:
        sweepId = args.sweepId
    else:
        sweepId = wandb.sweep(sweepConfig, entity=args.wandbEntity)
    # Multiple machines can run the same sweep
    print(f"### Sweep ID: {sweepId}")
    # The configs from wandb are updated with sweep config!
    wandb.agent(sweepId, function=run_sweep)





