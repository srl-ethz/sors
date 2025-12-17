#############################################################################
# PokeFlex Sim2Real evaluation script.
#
# Loads a YAML configuration to run one or many (trial, poke) sequences of the PokeFlex
# dataset. For each sequence it builds SimulationSettings/config, replays robot tool
# motion via a DiskContact constraint, simulates the deformable object, evaluates per-
# frame Chamfer distance against reconstructed meshes, and optionally exports VTU/PVD,
# renders frames, writes an MP4, and logs a CSV summary.
#############################################################################

import numpy as np
import csv
import yaml
import time
import sys
sys.path.append('./cpp/build')
import os
import glob
import re
import json
from py_sors import IO, SimulationSettings, get_disk_contact_constraint
from _renderer import export_mp4
from _utils import _read_and_transform_mesh, _chamfer_distance, _sample_on_triangles
from d03_pokeflex import PokeFlexEnv

#############################################################################
# Helper functions
#############################################################################

def load_cfg(path: str) -> dict:
    """Load YAML config into a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_settings_and_config(cfg: dict, trial: str, poke: str):
    """Given the full YAML cfg + a (trial, poke) pair, build:
      - SimulationSettings instance
      - config dict 
    """
    OBJECT = cfg["object"]
    TRIAL  = trial
    POKE   = poke
    TRIALPOKE = f"{TRIAL}{POKE}"
    
    ocfg = cfg["objectConfigTPRanges"][TRIALPOKE] # Frame range for this (trial, poke)
    initFrame, lastFrame = ocfg["initFrame"], ocfg["lastFrame"]
    config = dict(cfg["config"]) # Copy base config (so we can tweak per-trial without mutating cfg)

    # Paths
    paths = cfg["paths"]
    initial_mesh_root = paths["initialMeshRoot"].format(OBJECT=OBJECT)
    data_root         = paths["dataRoot"].format(OBJECT=OBJECT)
    output_root       = paths["outputRoot"]
    config["dataMeshFilePath"]      = f"{data_root}/{OBJECT}_{TRIAL}/meshes"
    config["dataForceFilePath"]     = f"{data_root}/{OBJECT}_{TRIAL}/robot_data.json"
    config["dataMeshFileInitFrame"] = initFrame

    # SimulationSettings
    sim_yaml = cfg["sim"]
    settings = SimulationSettings()
    settings.description = sim_yaml["description"]
    settings.verbose     = sim_yaml["verbose"]
    settings.numThreads  = sim_yaml["numThreads"]
    settings.dt       = (1.0 / config["frameRate"]) / max(1, sim_yaml["substeps"]) # Needs to be change for new substeps implementation
    settings.numSteps = lastFrame - initFrame + 1
    settings.CFLtimeSteppingFlag = sim_yaml["CFLtimeSteppingFlag"]
    settings.timeSteppingScheme  = sim_yaml["timeSteppingScheme"]
    settings.substeps            = sim_yaml["substeps"]
    settings.solverMethod        = sim_yaml["solverMethod"]
    settings.meshFilePath = f"{initial_mesh_root}/{TRIALPOKE}_mesh.msh"
    settings.meshFileName = os.path.basename(settings.meshFilePath)
    settings.meshType     = sim_yaml["meshType"]
    settings.outputFolder = os.path.join(output_root, f"{OBJECT}_{TRIALPOKE}") # Use a subfolder per TRIALPOKE (so outputs don’t collide)
    os.makedirs(settings.outputFolder, exist_ok=True)

    return settings, config

def load_robot_json(path: str):
    """Reads pokeflex robot_data.json with keys:
      - frame: string ID
      - forces: [Fx,Fy,Fz,Tx,Ty,Tz]
      - T_WT:   4x4 transform, last column is tip position (meters)
    Args:
        path: path to robot_data.json
    Returns:
        frame_ids: (N,) int array of frame numbers
        P_list: (N,3) tip positions in world frame [meter]
        F_list: (N,3) tip forces in world frame [Newton]
        Z_list: (N,3) local tool z-axis in world frame (unit vectors)
        T_list: (N,4,4) list of transforms
    """
    with open(path, "r") as f:
        data = json.load(f)

    frame_ids, P_list, F_list, T_list, Z_list = [], [], [], [], []
    for item in data:
        frame_ids.append(int(item["frame"]))
        F_list.append(np.asarray(item["forces"][:3], dtype=float))
        T = np.asarray(item["T_WT"], dtype=float)
        R = T[:3, :3]
        z_axis = R[:, 2] / (np.linalg.norm(R[:, 2]) + 1e-12)  # always take local z-axis
        P_list.append(T[:3, 3])
        T_list.append(T)
        Z_list.append(z_axis)
        
    return (np.array(frame_ids),
            np.vstack(P_list),
            np.vstack(F_list),
            np.vstack(Z_list),
            np.stack(T_list))
    
def apply_tool_offset(T_WT_all, tip_radius_m=0.005, axis="z"):
    """Apply a spherical tool tip offset to the TCP positions.
    Args:
        T_WT_all : (N,4,4) array of homogeneous transforms (world←tool)
        tip_radius_m : float, tool tip radius in meters
        axis : str, which tool axis points "forward" ('x','y','z')
               default 'z' → use R[:,2]
    Returns:
        P_contact : (N,3) array of contact points in world frame
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    ai = axis_map[axis]
    N = len(T_WT_all)
    positionContact = np.zeros((N, 3))
    for i in range(N):
        T = T_WT_all[i]
        R = T[:3, :3]
        positionCenter = T[:3, 3] # Tool center point
        forward = R[:, ai] / (np.linalg.norm(R[:, ai]) + 1e-12) # Rotated forward axis, normalized
        # Subtract radius along forward axis (flip sign if wrong)
        positionContact[i] = positionCenter - tip_radius_m * forward
    return positionContact

def transform_world_to_sim(pos_world, tx, ty, tz, scale):
    """ Transform world positions into sim/data frame,
    using the same (tx,ty,tz,scale) as your surface meshes.
    Args:
        p_world: (N,3) tip positions in world frame [meter]
        tx, ty, tz: translation applied to the mesh
        scale: scaling applied to the mesh
    Returns:
        P_sim: (N,3) tip positions in sim/data frame [meter]
    """
    T = np.array([tx, ty, tz])
    return (pos_world - T[None, :]) * float(scale)

#############################################################################
# Main simulation function
#############################################################################
    
def run_simulation (settings: SimulationSettings, config: dict):
    """Runs a single TRIALPOKE simulation with the provided settings/cfg.
    Args:
        - settings: SimulationSettings instance
        - config: dict with configuration parameters
    """
    # Initialize simulation environment 
    sim = PokeFlexEnv(settings, config)
    solution = sim.initialVertices.flatten() # Initial state
    initialFrame = config["dataMeshFileInitFrame"]
    
    # Get reconstructed mesh data paths (sorted)
    files = glob.glob(os.path.join(config["dataMeshFilePath"], "mesh-f*.obj"))
    dataMeshPaths = sorted(files, key=lambda x: int(re.search(r"mesh-f(\d+)\.obj", os.path.basename(x)).group(1)))

    if not dataMeshPaths:
        raise FileNotFoundError(f"No meshes found in {config['dataMeshFilePath']} matching 'mesh-f*.obj'")
    
    # Get force data from end-effector motion capture
    forceFrames, forceTipPosWorld, forceTipVectorsWorld, forceTipOrientation, forceTipTransformMatrices = load_robot_json(config["dataForceFilePath"])
    # Apply tool offset to force tip positions
    forceTipContactPosWorld = apply_tool_offset(forceTipTransformMatrices, tip_radius_m=config["toolOffset"], axis="z")
    forceTipContactPosWorldPlot = apply_tool_offset(forceTipTransformMatrices, tip_radius_m=config["toolOffset"] + config["toolOffsetPlot"], axis="z")  # For visualization only
    # Transform force tip positions and vectors to sim frame. (Robot tip positions are already in meters. No scaling needed.)
    forceTipPosWorld = transform_world_to_sim(forceTipPosWorld, sim.translationX, sim.translationY, sim.translationZ, 1.0)
    forceTipContactPosWorld = transform_world_to_sim(forceTipContactPosWorld, sim.translationX, sim.translationY, sim.translationZ, 1.0)
    forceTipContactPosWorldPlot = transform_world_to_sim(forceTipContactPosWorldPlot, sim.translationX, sim.translationY, sim.translationZ, 1.0)
    # Define disk from force tip positions and vectors
    diskCenter = forceTipContactPosWorld[initialFrame]
    diskNormal = forceTipOrientation[initialFrame]
    diskRadius = config["cylinderRadius"]
    # Grab and set DiskContact constraint (use correct index from constraint list)
    diskContactConstraint = get_disk_contact_constraint(sim.systemEnergy, 1)  # Index 1 because we have neumannBC first
    diskContactConstraint.set_disk(diskCenter, diskNormal, diskRadius, diskIdx=0)
    # Determine number of frames to evaluate
    framesToEval = min(len(dataMeshPaths), settings.numSteps)
    
    # Frame 0: sample surface points 
    simSurfacePts = _sample_on_triangles(solution.reshape(-1, 3), sim.surfaceFaces, n_samples=config["samplingNumberSim"], seed=config["samplingSeed"])
    # Frame 0: read reconstructed surface mesh with correct translation and scale
    verticesData, facesData, _ = _read_and_transform_mesh(dataMeshPaths[initialFrame], scale=sim.scale,
                                                            centerX=False, centerY=False, centerZ=False,
                                                            translationX=sim.translationX, translationY=sim.translationY, translationZ=sim.translationZ)
    
    # Save VTU of initial configuration
    if config["visualize"]: 
        IO.save_tet_VTU(f"{settings.outputFolder}/{settings.meshFileName}_0000", solution.reshape(-1,3), sim.elements)
    # Render initial configuration
    simulationMesh = None
    reconstructedMesh = None
    pointClouds = []
    disks = []
    if config["renderSim"]: simulationMesh = solution.reshape(-1,3)
    if config["renderData"]: reconstructedMesh = (verticesData, facesData)
    if config["renderRobotPos"]: pointClouds.append({"points": forceTipContactPosWorld[initialFrame].reshape(1,3), "radius": 0.003, "color": "00ff00"})  # Contact point (green)
    if config["renderSimSampling"]: pointClouds.append({"points": simSurfacePts, "radius": 0.0015, "color": "ff0000"})  # Sampled sim points (red)
    if config["renderRobotDisk"]: disks.append({"center": forceTipContactPosWorldPlot[initialFrame], "normal": forceTipOrientation[initialFrame], "radius": diskRadius, "color": "0000FF", "segments": 64})
    if config["render"]: sim.display(filename=f"{settings.outputFolder}/{settings.meshFileName}_0000.png", simulationMesh=simulationMesh, 
                                     reconstructedMesh=reconstructedMesh, pointClouds=pointClouds, disks=disks, spp = config["renderPrecision"])

    # Compute initial Chamfer distance
    chamferDistancesAll = []
    chamferDist = _chamfer_distance(simSurfacePts, verticesData) 
    chamferDistancesAll.append(chamferDist)
    
    # Store disk center, normal, chamfer distance (frame_idx, disk_cx, disk_cy, disk_cz, disk_nx, disk_ny, disk_nz, chamfer)
    storeData = []
    storeData.append([0, diskCenter[0], diskCenter[1], diskCenter[2], diskNormal[0], diskNormal[1], diskNormal[2], chamferDist])

    # Run simulation
    begin = time.time()
    for t in range(1, framesToEval):
        diskCenter = forceTipContactPosWorld[initialFrame + t]
        diskNormal = forceTipOrientation[initialFrame + t]  # Use orientation at frame t, we assume constant orientation during substeps
        # Update disk contact constraint based on end-effector position
        diskContactConstraint.set_disk(diskCenter, diskNormal, diskRadius, diskIdx=0)
        
        solution = sim.step(solution)
        
        # Evaluate Chamfer distance to reconstructed mesh
        simSurfacePts = _sample_on_triangles(solution.reshape(-1, 3), sim.surfaceFaces, n_samples=config["samplingNumberSim"], seed=config["samplingSeed"])
        verticesData, facesData, _ = _read_and_transform_mesh(dataMeshPaths[initialFrame + t], scale=sim.scale,
                                                                centerX=False, centerY=False, centerZ=False,
                                                                translationX=sim.translationX, translationY=sim.translationY, translationZ=sim.translationZ)
        chamferDist = _chamfer_distance(simSurfacePts, verticesData)
        chamferDistancesAll.append(chamferDist)
        print(f"Chamfer distance: {chamferDist:.6f} m")
        
        # Store disk center, normal, chamfer distance
        storeData.append([t, diskCenter[0], diskCenter[1], diskCenter[2], diskNormal[0], diskNormal[1], diskNormal[2], chamferDist])
        
        # Save VTU of current configuration
        if config["visualize"]:
            IO.save_tet_VTU(f"{settings.outputFolder}/{settings.meshFileName}_{t:04d}", solution.reshape(-1,3), sim.elements)
        # Render current configuration
        simulationMesh = None
        reconstructedMesh = None
        pointClouds = []
        disks = []
        if config["renderSim"]: simulationMesh = solution.reshape(-1,3)
        if config["renderData"]: reconstructedMesh = (verticesData, facesData)
        if config["renderRobotPos"]: pointClouds.append({"points": forceTipContactPosWorld[initialFrame + t].reshape(1,3), "radius": 0.003, "color": "00ff00"})  # Contact point (green)
        if config["renderSimSampling"]: pointClouds.append({"points": simSurfacePts, "radius": 0.0015, "color": "ff0000"})  # Sampled sim points (red)
        if config["renderRobotDisk"]: disks.append({"center": forceTipContactPosWorldPlot[initialFrame + t], "normal": forceTipOrientation[initialFrame + t], "radius": diskRadius, "color": "0000FF", "segments": 64})  # blue disk
        if config["render"]: sim.display(filename=f"{settings.outputFolder}/{settings.meshFileName}_{t:04d}.png", simulationMesh=simulationMesh, 
                    reconstructedMesh=reconstructedMesh, pointClouds=pointClouds, disks=disks,spp = config["renderPrecision"])        
    
        print(f"Time until simulation step {t}: {(time.time()-begin):.2f}s")
    
    print(f"Average time per simulation step: {1e3*(time.time()-begin)/settings.numSteps:.1f}ms")
    
    if config["visualize"]:
        IO.save_pvd(settings.meshFileName, settings.outputFolder, settings.numSteps, settings.dt)
    if config["render"]:
        export_mp4(f"{settings.outputFolder}", f"{settings.outputFolder}/{settings.meshFileName}.mp4", int(config["frameRate"]))
    if config["storeData"]:
        # write storeData to CSV
        csv_path = os.path.join(settings.outputFolder, f"{settings.meshFileName}_summary.csv")
        os.makedirs(settings.outputFolder, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame",
                "disk_cx", "disk_cy", "disk_cz",
                "disk_nx", "disk_ny", "disk_nz",
                "chamfer_m"
            ])
            writer.writerows(storeData)
        print(f"Saved summary CSV to: {csv_path}")
    print("================================================================================")
    print("================================================================================")
            
    return float(np.mean(chamferDistancesAll))

##############################################################################

def run_batch(cfg: dict):
    """Runs PokeFlex simulations according to cfg["mode"]:
    - mode == "all":    run all TRIALPOKEs from objectConfigTPRanges
    - mode == "single": run only the TRIAL+POKE specified in cfg["trial"], cfg["poke"]
    """
    OBJECT = cfg["object"]
    TPRanges = cfg["objectConfigTPRanges"]
    mode = cfg.get("mode", "all")  # Default to "all" if missing
    chamferDistanceMeanAll = []
    
    if mode == "single":
        trial = cfg["trial"]
        poke  = cfg["poke"]
        trialpoke = f"{trial}{poke}"
        if trialpoke not in TPRanges:
            raise ValueError(
                f"Requested single mode {trialpoke} not in objectConfigTPRanges."
            )
        tp_items = [(trialpoke, TPRanges[trialpoke])]
    elif mode == "all":
        tp_items = list(TPRanges.items())
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'all' or 'single'.")
    
    for TRIALPOKE, fr in tp_items:
        TRIAL = TRIALPOKE[:2]
        POKE  = TRIALPOKE[2:]

        settings, config = build_settings_and_config(cfg, TRIAL, POKE) # Build per-(trial,poke) settings and config

        print(f"Running {OBJECT} / {TRIALPOKE}")
        print(f"  Reconstructed Meshes:  {config['dataMeshFilePath']}")
        print(f"  Youngs modulus:        {config['youngsModulus']} Pa, alpha: {config['alpha']} s^(-1), poissonsRatio: {config['poissonsRatio']}")
        print(f"  Mesh:                  {settings.meshFilePath}")
        print(f"  Output:                {settings.outputFolder}")

        try:
            chamferDistanceMean = run_simulation(settings, config) 
            chamferDistanceMeanAll.append(chamferDistanceMean)
        except Exception as e:
            print(f"Simulation {OBJECT} / {TRIALPOKE} failed! {e}")
            print("Added large penalty to chamfer distance.") # Large penalty if simulation fails
            chamferDistanceMeanAll.append(1)
            break  # If one simulation fails, stop the optimization here

    log = {"chamferDistanceMeanAll": chamferDistanceMeanAll}
    return log
    
    
if __name__ == "__main__":
    root = os.path.dirname(__file__)
    cfg_path = os.path.join(root, "configs", "config_pokeflex_sim2real.yml")
    cfg = load_cfg(cfg_path)
    run_batch(cfg)