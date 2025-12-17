import gmsh

# Initialize Gmsh
gmsh.initialize()

# Create a new model
gmsh.model.add("cylinder")

# Define cylinder parameters
radius = 0.1     # Radius of the cylinder
length = 0.5     # Length of the cylinder (along the Z-axis)
center = [0, 0, 0]  # Base of the cylinder starts at Z = 0

# Create the cylinder (a volume)
gmsh.model.occ.addCylinder(center[0], center[1], center[2],  # Base position
                           length, 0, 0,  # Vector for height
                           radius)  # Radius

# Synchronize the CAD representation with the Gmsh model
gmsh.model.occ.synchronize()

# Define mesh resolution
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", radius / 2.5)  # Adjust for finer mesh
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", radius / 2.5)

# Generate the tetrahedral (3D) mesh
gmsh.model.mesh.generate(3)

# Save the mesh to a file
gmsh.write("cylinder.msh")

# Finalize Gmsh
gmsh.finalize()