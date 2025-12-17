################################################################################
# Run hyperparameter optimization for Pokeflex Poking using Weights & Biases 
# (wandb) to run Bayesian Optimization.
# - Configurations should be defined in config_pokeflex_sim2real.yml
# - Hyperparameters to optimize should be defined in the config_pokeflex_opt.yml
################################################################################

import sys, traceback, argparse, yaml
from datetime import datetime
import wandb
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
sys.path.append('./cpp/build')
from d03a_pokeflex_sim2real import load_cfg, run_batch  

# Sweep config initialization
configs = {
    "youngsModulus":    1e3,
    "alpha":            10.0,
    "poissonsRatio":     0.3,
}  

def run_sweep ():
    """ Run a single sweep iteration with wandb.     
    Calls run_simulation_all with the current hyperparameters from wandb.
    Args:
        simulationConfig (dict): Configuration dictionary for the pokeflex simulation.
    """
    # Load simulation config (config_pokeflex_sim2real.yml)
    root = os.path.dirname(__file__)
    cfg_path = os.path.join(root, "configs", "config_pokeflex_sim2real.yml")
    simulationConfig = load_cfg(cfg_path)
    simulationConfig['config']['mode'] = 'all'       # Overwrite mode to 'all' for hyperparameter optimization
    simulationConfig['config']['visualize'] = False  # Disable visualization for optimization runs
    simulationConfig['config']['render'] = False     # Disable rendering for optimization runs
    
    try:
        wandb.init(config=configs)

        ### Overwrite hyperparameters from wandb 
        simulationConfig['config']['youngsModulus'] = wandb.config.get("youngsModulus")
        simulationConfig['config']['alpha'] = wandb.config.get("alpha")
        simulationConfig['config']['poissonsRatio'] = wandb.config.get("poissonsRatio")
        
        # log = run_simulation_all(settings, config = pokeflex_config, youngsModulus=youngsModulus, alpha=alpha, poissonsRatio=poissonsRatio)
        log = run_batch(simulationConfig)

        wandb.log({
            "error_mean": np.mean(log["chamferDistanceMeanAll"]),
            "error_std": np.std(log["chamferDistanceMeanAll"]),
        })

    except:
        # Exit gracefully, so wandb logs the problem
        print(traceback.print_exc())
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Optimization for Pokeflex Poking")
    parser.add_argument('--sweepId', type=str, default=None, help='Sweep ID to continue running previous WandB sweep')
    parser.add_argument('--sweepConfig', type=str, default="python/configs/config_pokeflex_opt.yml", help='Config file for hyperparameter sweep')
    parser.add_argument('--wandbEntity', type=str, default="", help='WandB sweep entity.')
    args = parser.parse_args()

    # Load optimization config (config_pokeflex_opt.yml)
    with open(args.sweepConfig, 'r', encoding='utf-8') as stream:
        try:
            sweepConfig = yaml.safe_load(stream)
            print(sweepConfig)
        except yaml.YAMLError as exc:
            print(exc)
    # Name the sweep based on current date
    sweepConfig['name'] = f"sweep_pokeflex_{datetime.now().strftime('%Y%m%d')}"
    
    # Continue running existing sweeps
    if args.sweepId is not None:
        sweepId = args.sweepId
    else:
        sweepId = wandb.sweep(sweepConfig, entity=args.wandbEntity)
    # Multiple machines can run the same sweep
    print(f"### Sweep ID: {sweepId}")
    # The configs from wandb are updated with sweep config!
    wandb.agent(sweepId, function=run_sweep)





