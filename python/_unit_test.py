################################################################################
# Test Functions for Github Workflow through Pytest.
################################################################################

import os
import sys
import shutil
sys.path.append('./cpp/build')
import pytest
from py_sors import SimulationSettings
from d01_msd_timestepping import main as main_01
from d02_cantilever import main as main_02
from d03_pokeflex import main as main_03
from d04_sopra import main as main_04
from d05_hopping_leg import main as main_05


@pytest.fixture(autouse=True)
def run_around_tests ():
    # Code that will run before tests
    if not os.path.exists("tmp"): os.makedirs("tmp")

    # A test function will be run at this point
    yield

    # Code that will run after tests
    shutil.rmtree("tmp")
    

def test_01_msd ():
    # Simulation settings
    settings = SimulationSettings()
    settings.description = "Demo of a mass-spring damper with external forces under gravity and adaptive time stepping."
    settings.meshFilePath = "cpp/demos/meshes/msd_cylinder_v3.msh"
    settings.outputFolder = "tmp"
    settings.meshFileName = settings.meshFilePath.split("/")[-1]
    settings.meshType = "Tetrahedron"
    settings.dt = 1e-2
    settings.numSteps = 5
    settings.CFLtimeSteppingFlag = True
    settings.timeSteppingScheme = "crank_nicolson"
    settings.solverMethod = "minimize_newton"
    settings.verbose = 0
    settings.numThreads = 1
    # settings.verbose_print()
    main_01(settings)
    assert True, "Demo 01 MSD Timestepping ran successfully."

def test_d02_cantilever ():
    # Simulation settings
    settings = SimulationSettings()
    settings.description = "Demo of passive cantilever."
    settings.meshFilePath = "cpp/demos/meshes/cantilever.msh"
    settings.outputFolder = "tmp"
    settings.meshFileName = settings.meshFilePath.split("/")[-1]
    settings.meshType = "Tetrahedron"
    settings.dt = 1e-2
    settings.numSteps = 5
    settings.substeps = 2
    settings.CFLtimeSteppingFlag = False
    settings.timeSteppingScheme = "backward_euler"
    settings.solverMethod = "minimize_newton"
    settings.verbose = 0
    settings.numThreads = 1
    # settings.verbose_print()
    main_02(settings)
    assert True, "Demo 02 Cantilever ran successfully."
    
    
def test_d03_pokeflex ():
    # Simulation settings
    settings = SimulationSettings()
    settings.description = "Demo of a deformable cube being poked by robot arm."
    settings.meshFilePath = "cpp/demos/meshes/pokeflexFoamDice/T1P2_mesh.msh"
    settings.outputFolder = "tmp"
    settings.meshFileName = settings.meshFilePath.split("/")[-1]
    settings.meshType = "Tetrahedron"
    settings.dt = 1e-2
    settings.numSteps = 3
    settings.substeps = 1
    settings.CFLtimeSteppingFlag = False
    settings.timeSteppingScheme = "crank_nicolson"
    settings.solverMethod = "minimize_SQP"
    settings.verbose = 0
    settings.numThreads = 1
    # Additional configuration parameters
    config = {
        "objectWeight": 0.14,                   # weight of the object in kg
        "youngsModulus": 1345.60,               # Young's modulus in Pa
        "poissonsRatio": 0.0177,                # Poisson's ratio
        "alpha": 170,                           # mass damping coefficient
        "fixingGround": 0.002,                  # y height below which vertices are fixed to simulate ground contact (in m)
        "cylinderRadius": 0.02,                 # radius of the cylinder end-effector (in m)
        "visualize": False,                      # whether to save VTU and PVD files for visualization
        "render": False,                         # whether to render images using PBRT
        "renderPrecision": 4,                   # PBRT samples per pixel
        "frameRate": 30,                        # frame rate for the output video
        "tipBaseCenter": [0.0, 0.25, 0.0],      # x, y, z base position in meters
        "tipAmplitude": 0.17,                   # amplitude (m) of up/down motion
    }
    # settings.verbose_print()
    main_03(settings, config)
    assert True, "Demo 03 Pokeflex ran successfully."
    
    
def test_d04_sopra ():
    # Simulation settings
    settings = SimulationSettings()
    settings.description = "Demo of pressure actuated 3D arm (SoPrA)."
    settings.meshFilePath = "cpp/demos/meshes/arm.msh"
    settings.outputFolder = "tmp"
    settings.meshFileName = settings.meshFilePath.split("/")[-1]
    settings.meshType = "Tetrahedron"
    settings.dt = 1e-2
    settings.numSteps = 5
    settings.substeps = 10
    settings.CFLtimeSteppingFlag = False
    settings.timeSteppingScheme = "backward_euler"
    settings.solverMethod = "minimize_newton"
    settings.verbose = 0
    settings.numThreads = 1
    # settings.verbose_print()
    main_04(settings)
    assert True, "Demo 04 Sopra ran successfully."


def test_d05_hopping_leg ():
    # Simulation settings
    settings = SimulationSettings()
    settings.description = "Demo of a hopping leg that is muscle-actuated."
    settings.meshFilePath = "cpp/demos/meshes/hopping_leg.msh"
    settings.outputFolder = "tmp"
    settings.meshFileName = settings.meshFilePath.split("/")[-1]
    settings.meshType = "Hexahedron"
    settings.dt = 2e-2/25
    settings.numSteps = 3
    settings.CFLtimeSteppingFlag = False
    settings.timeSteppingScheme = "backward_euler"
    settings.solverMethod = "minimize_SQP"
    settings.verbose = 0
    settings.numThreads = 1
    # settings.verbose_print()
    main_05(settings)
    assert True, "Demo 05 Hopping Leg ran successfully."