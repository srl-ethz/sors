import gmsh
import math

# Initialize Gmsh
gmsh.initialize()

# Create a new model
gmsh.model.add("doublePendulum")

# Define the base parameters
base_origin = [0, 0, 0]  # Base starts at the origin
base_length = 1       # 1m
base_width = 0.2       # 0.2m
base_height = 0.05     # 0.05m

# Create the base (a box)
base = gmsh.model.occ.addBox(base_origin[0], base_origin[1], base_origin[2], 
                             base_length, base_width, base_height)

# Define pendulum parameters
pendulum_side = 0.05    # 0.05m side length of the quadratic cross-section
pendulum_length = 0.6   # 0.6m length of the pendulums
pendulum_z_offset = 0  # Pendulums start at the bottom of the base

# First pendulum position (attached at 10 mm along the base)
pendulum1_x = 0.1 - pendulum_side / 2  
pendulum1_y = (base_width - pendulum_side) / 2  # Centered in the y-direction
pendulum1 = gmsh.model.occ.addBox(pendulum1_x, pendulum1_y, pendulum_z_offset, 
                                  pendulum_side, pendulum_side, -pendulum_length)

# Second pendulum position (attached at 90 mm along the base)
pendulum2_x = 0.9 - pendulum_side / 2
pendulum2_y = (base_width - pendulum_side) / 2  # Centered in the y-direction
pendulum2 = gmsh.model.occ.addBox(pendulum2_x, pendulum2_y, pendulum_z_offset, 
                                  pendulum_side, pendulum_side, -pendulum_length)

cube_side = 0.1  # 0.1m side length of cube

# First cube position (at the bottom of the first pendulum)
cube1_x = pendulum1_x - (cube_side - pendulum_side) / 2
cube1_y = pendulum1_y - (cube_side - pendulum_side) / 2
cube1_z = pendulum_z_offset - pendulum_length - cube_side
cube1 = gmsh.model.occ.addBox(cube1_x, cube1_y, cube1_z, 
                              cube_side, cube_side, cube_side)

# Second cube position (at the bottom of the second pendulum)
cube2_x = pendulum2_x - (cube_side - pendulum_side) / 2
cube2_y = pendulum2_y - (cube_side - pendulum_side) / 2
cube2_z = pendulum_z_offset - pendulum_length - cube_side
cube2 = gmsh.model.occ.addBox(cube2_x, cube2_y, cube2_z, 
                              cube_side, cube_side, cube_side)

# Fuse the base, pendulums, and cubes into a single volume
gmsh.model.occ.fuse([(3, base)], [(3, pendulum1), (3, pendulum2), (3, cube1), (3, cube2)])

# Synchronize the internal CAD representation with the Gmsh model
gmsh.model.occ.synchronize()

# Generate the 3D mesh
gmsh.model.mesh.generate(3)

# Optionally, save the mesh to a file
gmsh.write("doublePendulumv2.msh")

# Finalize the Gmsh session
gmsh.finalize()


