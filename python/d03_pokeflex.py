##############################################################################
# PokeFlex Simulation Environment.

# Builds a tetrahedral soft-body simulation with a disk-based contact end-effector and
# runs a short poking rollout with optional VTU/PVD export and PBRT rendering.
##############################################################################

import numpy as np
import time
import sys
sys.path.append('./cpp/build')
import os
import json
from py_sors import Params, Energy, Solver, IO, SimulationSettings, get_disk_contact_constraint
from _renderer import PbrtRenderer, export_mp4
from _utils import _sample_on_triangles, _read_and_transform_mesh, _make_disk_mesh, _compute_stats

################################################################################
# Simulation settings
settings = SimulationSettings()
settings.description = "Demo of cylinder poking a soft foam dice."
settings.meshFilePath = "cpp/demos/meshes/pokeflexFoamDice/T1P2_mesh.msh"
settings.outputFolder = "output/pokeflex_demo"
settings.meshFileName = settings.meshFilePath.split("/")[-1]
settings.meshType = "Tetrahedron"
settings.dt = 1.0/30.0  
settings.numSteps = 50
settings.substeps = 1
settings.CFLtimeSteppingFlag = False
settings.timeSteppingScheme = "crank_nicolson" # "backward_euler", "crank_nicolson"
settings.solverMethod = "minimize_SQP"
settings.verbose = True
settings.numThreads = 8

# Additional configuration parameters
config = {
    "objectWeight": 0.14,                   # Weight of the object in kg
    "youngsModulus": 1345.60,               # Young's modulus in Pa
    "poissonsRatio": 0.0177,                # Poisson's ratio
    "alpha": 170,                           # Mass damping coefficient
    "fixingGround": 0.002,                  # y-height below which vertices are fixed to simulate ground contact (in m)
    "cylinderRadius": 0.02,                 # Radius of the cylinder end-effector (in m)
    "visualize": True,                      # Whether to save VTU and PVD files for visualization
    "render": True,                         # whether to render images using PBRT
    "renderPrecision": 4,                   # PBRT samples per pixel
    "frameRate": 30,                        # Frame rate for the output video
    "tipBaseCenter": [0.0, 0.25, 0.0],      # x, y, z base position in meters
    "tipAmplitude": 0.17,                   # Amplitude (m) of up/down motion
}
################################################################################

class PokeFlexEnv:
    """PokeFlex poking environment for soft tetrahedral objects.

    Loads and recenters a tetrahedral mesh (mm → m), identifies surface vertices/faces,
    computes mesh volume to derive density from object weight, and instantiates 
    energies/constraints (Neo-Hookean elasticity, gravity, damping, fixed-ground BC, and
    a DiskContact constraint used as the end-effector).
    """
    def __init__(self, settings: SimulationSettings, config: dict) -> None:
        # 1) Read Mesh #########################################################
        # Scale 1e-3 to convert mm to m
        if not os.path.exists(settings.meshFilePath):
            raise FileNotFoundError(f"Mesh file not found: {settings.meshFilePath}")
        vertices, elements, transformInfo = _read_and_transform_mesh(settings.meshFilePath, scale=1e-3,
                                                                        centerX=True, centerY=False, centerZ=True,
                                                                        translationX=0.0, translationY=0.0, translationZ=0.0)
        # Move mesh above ground (y=0)
        ymin = np.min(vertices[:,1])
        vertices[:,1] -= ymin  # Translate so that lowest point is at y=0
                
        self.initialVertices = vertices.copy()  # Store initial vertices for reference
        self.vertices = vertices  
        self.elements = elements
        # transformInfo = (scale, tx, ty, tz)
        self.scale = transformInfo[0]  # Scale factor that was applied to the mesh
        self.translationX = transformInfo[1]  
        self.translationY = transformInfo[2] + ymin  
        self.translationZ = transformInfo[3] 
        
        # Get surface faces and surface vertex indices
        surfaceGroups, _ = IO.extract_tet_surfaces(vertices, elements)
        # Flatten surface groups into single array
        surfaceFaces = np.vstack(surfaceGroups)  # (K, 3)
        surfaceVertexIdx = np.unique(surfaceFaces) # (NS,) number of surface vertices
        self.surfaceFaces = surfaceFaces # Need this for sampling surface points
        self.surfaceVertexIdx = surfaceVertexIdx # Need this for contact constraints
       
        # 2) Compute and cache mesh stats ######################################
        os.makedirs(settings.outputFolder, exist_ok=True)
        cache_path = os.path.join(
            settings.outputFolder,
            f"{os.path.splitext(settings.meshFileName)[0]}_meta.json"
        )
        stats = _compute_stats(self.vertices, self.elements)   
        stats["scale_applied"] = 1e-3
        stats["mesh_path"] = settings.meshFilePath
        with open(cache_path, "w") as f:
            json.dump(stats, f, indent=2)
        self.volume = stats["volume_m3"]
        self.density = config["objectWeight"] / self.volume  # density in kg/m^3

        # 3) Define Energies and constraints ###################################
        elementEnergiesList = []
        elementParameterList = []
        for j in range(self.elements.shape[0]):
            elementEnergiesList.append(set(["stableneohookean"])) 
            elementParameterList.append(Params(["density", "youngsModulus", "poissonsRatio"], [self.density, config["youngsModulus"], config["poissonsRatio"]]))
        gravAcceleration = np.array([0.0, -9.81, 0.0]) # Gravitational field in y direction
        
        constraintTypesList = ['neumannBC', 'diskContact']  # We will use neumannBC to fix bottom vertices, and diskContact for contact with end-effector
        # Boundary constraints
        neumannBCmask = np.zeros(3*self.vertices.shape[0])
        neumannBCvalue = np.zeros(3*self.vertices.shape[0])
        fixedVerticesIdx = []
        for i in range(self.vertices.shape[0]):
            if self.vertices[i,1] < config["fixingGround"]: 
                neumannBCmask[3*i] = 1;     neumannBCvalue[3*i] = 0.0
                neumannBCmask[3*i+1] = 1;   neumannBCvalue[3*i+1] = 0.0
                neumannBCmask[3*i+2] = 1;   neumannBCvalue[3*i+2] = 0.0
                fixedVerticesIdx.append(i)        
        print(f"  Fixed {len(fixedVerticesIdx)} vertices on the bottom (y direction) to simulate contact with the ground.")
        # Contact constraint
        diskCenter = np.array([0.0, 0.0, 0.0])  # Will be updated based on end-effector position
        diskNormal = np.array([0.0, -1.0, 0.0])  # Will be updated based on end-effector orientation
        diskRadius = config["cylinderRadius"]  
        diskCenterVectorList = diskCenter.tolist()
        diskNormalVectorList = diskNormal.tolist()
        diskRadiusList = [diskRadius]

        constraintParameterList = Params(["neumannBCmask", "neumannBCvalue", "diskCenterVectorList", "diskNormalVectorList", "diskRadiusList", "surfaceVerticesIdx"], 
                                         [neumannBCmask, neumannBCvalue, diskCenterVectorList, diskNormalVectorList, diskRadiusList, surfaceVertexIdx.tolist()])  
        forceTypesList = []
        forceParameterList = Params()  
        
        # 4) Initialize energy and solver ######################################
        self.systemEnergy = Energy(self.vertices, self.elements, settings, gravAcceleration, config["alpha"], elementEnergiesList, elementParameterList, constraintTypesList, constraintParameterList, forceTypesList, forceParameterList)
        self.solver = Solver(settings)

    ############################################################################

    def step (self, solution, actuation = None):
        """Advance the simulation by one time step."""
        if actuation is None:
            return self.solver.step(solution, self.systemEnergy)
        else:
            return self.solver.step(solution, self.systemEnergy, actuation)
    
    ############################################################################
    
    def display(self, filename: str, 
                simulationMesh: np.ndarray | None, 
                reconstructedMesh: tuple[np.ndarray, np.ndarray] | None,
                pointClouds: list[dict] | None,
                disks: list[dict] | None,
                spp: int = 4):
        """Render the current tet mesh, optional external surface mesh, and point 
        clouds with individual radii/colors, and disks to represent end-effector.
        Args:
            filename: output image path (.png or .exr)
            simulationMesh: (N,3) simulated tet vertices
            reconstructedMesh: tuple (surfaceVertices, surfaceFaces) where
                - surfaceVertices: (Ns,3) float, triangle-vertex positions 
                - surfaceFaces: (Ks,3) int, triangle indices 
            pointClouds: list of dicts 
                - "points": (M,3) array OR list/tuple of points
                - "radius": float (m), default 0.002
                - "color":  hex str, default "ff0000"
            disks: optional list of dicts, each with:
                - "center": (3,) array
                - "normal": (3,) array
                - "radius": float
                - "segments": int (optional, default 64)
                - "color": hex str (optional, default "00aa00")
            spp: PBRT samples per pixel
        """
        options = {
            'file_name': filename,
            'light_map': 'uffizi-large.exr',
            'sample': spp,
            'max_depth': 2,
            'camera_pos': (-0.2, 0.75, 0.5),
            'camera_lookat': (0, 0, 0.1)     
        }
        transforms=[
                ('s', 1.0),
                ('t', [0.0, 0.0, 0.0]),
                ('r', [np.pi/2, 1.0, 0.0, 0.0])  # Rotate into y-up frame
        ]
        renderer = PbrtRenderer(options)
         # 1) Simulated tet mesh
        if simulationMesh is not None:
            renderer.add_tri_mesh(vertices=simulationMesh, elements=self.elements, render_edges=True, color="496d8a", transforms=transforms)
        # 2) Reconstructed surface mesh
        if reconstructedMesh is not None and len(reconstructedMesh) == 2:
            surfaceVertices, surfaceFaces = reconstructedMesh
            renderer.add_tri_mesh(vertices=surfaceVertices, elements=surfaceFaces, render_edges=False, color="ffcc00", transforms=transforms)
        # 3) Point clouds 
        if pointClouds is not None:
            for cloud in pointClouds:
                pts = np.asarray(cloud["points"], dtype=float).reshape(-1, 3)
                radius = float(cloud.get("radius", 0.002))
                color  = cloud.get("color", "ff0000")
                for p in pts:
                    shape_info = {'name': 'sphere', 'radius': radius, 'center': p}
                    renderer.add_shape_mesh(shape_info, transforms=transforms, color=color)      
        # 4) Disks (end-effector)
        if disks is not None:
            for d in disks:
                center  = np.asarray(d["center"], float)
                normal  = np.asarray(d["normal"], float)
                radius  = float(d["radius"])
                segs    = int(d.get("segments", 64))
                color   = d.get("color", "00aa00")
                Vd, Fd  = _make_disk_mesh(center, normal, radius, segs)
                renderer.add_tri_mesh(vertices=Vd, elements=Fd,
                                    render_edges=False, color=color, transforms=transforms)
        # 5) Ground
        renderer.add_tri_mesh(objFile='asset/mesh/curved_ground.obj',
                            texture_img='chkbd_24_0.7',
                            transforms=[('s', 4)])

        renderer.render()

################################################################################
    
def main (settings, config = config) -> None:
    """Run a poking rollout with a sinusoidally moving disk end-effector.

    Builds a PokeFlexEnv, generates a vertical (y-axis) sinusoidal tool trajectory,
    updates the DiskContact constraint each frame, steps the simulation, and optionally
    exports VTU/PVD files and PBRT-rendered frames + an MP4.
    """
    # Initialize simulation environment 
    sim = PokeFlexEnv(settings, config)
    solution = sim.initialVertices.flatten() # Initial state
    
    # sinusoidal tool motion
    numFrames = settings.numSteps
    dtFrame   = settings.dt   
    initialFrame = 0
    totalFrames = numFrames + initialFrame
    baseCenter = np.array(config["tipBaseCenter"])
    robotMovementAmplitude = config["tipAmplitude"]
    forceTipContactPosWorld = []
    forceTipOrientation = []
    for frame in range(totalFrames):
        timePhys = frame * dtFrame
        # Sinusoidal up/down motion in y direction
        yBase = baseCenter[1] - robotMovementAmplitude * np.sin(np.pi * timePhys / (numFrames * dtFrame))
        center = np.array([baseCenter[0], yBase, baseCenter[2]])
        forceTipContactPosWorld.append(center)
        forceTipOrientation.append(np.array([0.0, -1.0, 0.0]))  # Pointing downwards
    forceTipContactPosWorld = np.asarray(forceTipContactPosWorld)
    forceTipOrientation = np.asarray(forceTipOrientation)
    
    # Define disk from force tip positions and vectors
    diskCenter = forceTipContactPosWorld[initialFrame]
    diskNormal = forceTipOrientation[initialFrame]
    diskRadius = config["cylinderRadius"]
    
    # Grab and set DiskContact constraint (use correct index from constraint list)
    diskContactConstraint = get_disk_contact_constraint(sim.systemEnergy, 1)  # Index 1 because we have neumannBC first, get_disk_contact_constraint lives in py_sors
    diskContactConstraint.set_disk(diskCenter, diskNormal, diskRadius, diskIdx=0)
    
    # Save VTU of initial configuration
    if config["visualize"]: 
        IO.save_tet_VTU(f"{settings.outputFolder}/{settings.meshFileName}_0000", solution.reshape(-1,3), sim.elements)
    # Render initial configuration
    simulationMesh = None
    disks = []
    if config["render"]: 
        simulationMesh = solution.reshape(-1,3)
        disks.append({"center": forceTipContactPosWorld[initialFrame], "normal": forceTipOrientation[initialFrame], "radius": diskRadius, "color": "0000FF", "segments": 64})
        sim.display(filename=f"{settings.outputFolder}/{settings.meshFileName}_0000.png", simulationMesh=simulationMesh, 
                                     reconstructedMesh=None, pointClouds=None, disks=disks, spp = config["renderPrecision"])
    # Run simulation
    begin = time.time()
    for t in range(1, settings.numSteps):
        diskCenter = forceTipContactPosWorld[initialFrame + t]
        diskNormal = forceTipOrientation[initialFrame + t]  
        # Update disk contact constraint based on end-effector position
        diskContactConstraint.set_disk(diskCenter, diskNormal, diskRadius, diskIdx=0)
        
        solution = sim.step(solution)
        
        # Save VTU of current configuration
        if config["visualize"]:
            IO.save_tet_VTU(f"{settings.outputFolder}/{settings.meshFileName}_{t:04d}", solution.reshape(-1,3), sim.elements)
        # Render current configuration    
        simulationMesh = None
        disks = []
        if config["render"]: 
            simulationMesh = solution.reshape(-1,3)
            disks.append({"center": forceTipContactPosWorld[initialFrame + t], "normal": forceTipOrientation[initialFrame + t], "radius": diskRadius, "color": "0000FF", "segments": 64})
            sim.display(filename=f"{settings.outputFolder}/{settings.meshFileName}_{t:04d}.png", simulationMesh=simulationMesh, 
                                        reconstructedMesh=None, pointClouds=None, disks=disks, spp = config["renderPrecision"])   
        
        print(f"Time until simulation step {t}: {(time.time()-begin):.2f}s")
    
    print(f"Average time per simulation step: {1e3*(time.time()-begin)/settings.numSteps:.1f}ms")
    
    if config["visualize"]:
        IO.save_pvd(settings.meshFileName, settings.outputFolder, settings.numSteps, settings.dt)
    if config["render"]:
        export_mp4(f"{settings.outputFolder}", f"{settings.outputFolder}/{settings.meshFileName}.mp4", int(config["frameRate"]))
    print("================================================================================")
        
        
if __name__ == "__main__":
    main(settings, config)