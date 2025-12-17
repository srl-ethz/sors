import meshio
import numpy as np

# Calculate side length for the equilateral base triangle
side_length = 0.1414

# Define the vertices (nodes) of the bipyramid
# The base triangle is centered at z = 0.5 with side length 0.1414
# Each tetrahedron extends 0.1 units above and below the center
points = np.array([
    [0, side_length / 2, 0.5],                  # Base vertex 1
    [-side_length * np.sqrt(3) / 4, -0.5 * side_length / 2, 0.5],  # Base vertex 2
    [side_length * np.sqrt(3) / 4, -0.5 * side_length / 2, 0.5],   # Base vertex 3
    [0, 0, 0.6],                               # Top vertex, 0.1 units above the center
    [0, 0, 0.4]                                # Bottom vertex, 0.1 units below the center
])

# Define the elements (tetrahedrons) using the points
# Each tetrahedron is defined by four vertex indices (0-based)
cells = [
    ("tetra", np.array([
        [0, 1, 2, 3],  # Top tetrahedron
        [0, 1, 2, 4]   # Bottom tetrahedron
    ]))
]

# Create a mesh object and write to an .msh file
mesh = meshio.Mesh(points=points, cells=cells)
meshio.write("bipyramid.msh", mesh, file_format="gmsh")