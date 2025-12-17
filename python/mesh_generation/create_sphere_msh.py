import gmsh
import math

# Initialize Gmsh
gmsh.initialize()

# create a new model
gmsh.model.add("sphere")

# Define the sphere parameters
center = [0, 0, 0.5]  # Center of the sphere at z = 1m
radius = 0.1         # Radius of the spheres (0.1 m = 10 cm)

# Create the sphere (a volume)
gmsh.model.occ.addSphere(center[0], center[1], center[2], radius)

# Synchronize the internal CAD representation with the gmsh model
gmsh.model.occ.synchronize()

# Define a finer mesh size
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", radius / 7)  # Decrease for finer mesh
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", radius / 7)

# Generate 3D mesh
gmsh.model.mesh.generate(3)

# Optionally, save the mesh to a file
gmsh.write("sphere3.msh")

# Finalize the gmsh session
gmsh.finalize()

