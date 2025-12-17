# Utility script to clean unused vertices from a Gmsh tetrahedral mesh.
#
# The script:
#   - Loads a mesh file given as a command-line argument,
#   - Removes unreferenced vertices by remapping node indices used by tetrahedra,
#   - Writes a cleaned mesh with only tetrahedral elements to `<name>_cleaned.msh`,
#   - Computes a mapping from each vertex to the physical volume tags it belongs to,
#   - Writes this node-to-volume-tag mapping to `<name>_tags.txt`.

import sys
import pathlib
import numpy
import meshio
in_path = pathlib.Path(sys.argv[1])

# Cleaning mesh
mesh = meshio.read(in_path)
points = mesh.points
print("# points before cleaning: ", points.shape)
point_map = {}

# Get tetrahedral element types from cells
print(f"Mesh contains {len(mesh.cells)} cell blocks.")
tetIdx = []
for i, cell in enumerate(mesh.cells):
    print(f"Cell type: {cell.type}, Number of cells: {len(cell.data)}")
    if cell.type == "tetra":
        print("Found tetrahedral elements.")
        tetIdx.append(i)

if len(tetIdx) == 0:
    raise ValueError("No tetrahedral elements found in the mesh.")
elif len(tetIdx) > 1:
    raise ValueError("Multiple tetrahedral element blocks found in the mesh. Please ensure only one block of tetrahedral elements is present.")
tets = mesh.cells[tetIdx[0]].data

for tet in tets:
    for i, vi in enumerate(tet):
        if vi not in point_map:
            point_map[vi] = len(point_map)
        tet[i] = point_map[vi]

points = numpy.empty((len(point_map), 3))
for k, v in point_map.items():
    points[v] = mesh.points[k]

mesh.points = points
mesh.cells[tetIdx[0]].data = tets

# Only keep tetrahedral elements and their associated cell data
mesh.cells = [mesh.cells[tetIdx[0]]]
mesh.cell_data = {key: [mesh.cell_data[key][tetIdx[0]]] for key in mesh.cell_data}

print("# points after cleaning: ", points.shape)

# Generate the output file path with "_cleaned.msh" appended
out_path = in_path.with_name(in_path.stem + "_cleaned.msh")

# Write the cleaned mesh to the new file
meshio.write(out_path, mesh, file_format="gmsh22")

print(f"Cleaned mesh written to: {out_path}")

# Mapping nodes to volume tags
def map_nodes_to_volume_tags(mesh):
    node_volume_tags = {node_id: set() for node_id in range(len(mesh.points))}

    # Loop through each cell block
    for cell_block, cell_data in zip(mesh.cells, mesh.cell_data["gmsh:physical"]):
        if cell_block.type in ["tetra", "hexahedron"]:  # Considering tetrahedra and hexahedra
            for cell_index, cell in enumerate(cell_block.data):
                volume_tag = cell_data[cell_index]
                for node_id in cell:
                    node_volume_tags[node_id].add(volume_tag)

    # Converting sets to sorted lists for consistency
    node_volume_tags = {node_id: sorted(list(tags)) for node_id, tags in node_volume_tags.items()}

    return node_volume_tags

node_volume_tags = map_nodes_to_volume_tags(mesh)

# Display the volume tags for each vertex
for node_id, volume_tags in node_volume_tags.items():
    print(f"Node {node_id}: Volume Tags {volume_tags}")

# Write the volume tags to a text file
# Generate the output file path with "_cleaned.msh" appended
out_path = in_path.with_name(in_path.stem + "_tags.txt")

with open(out_path, 'w') as f:
    for node_id, volume_tags in node_volume_tags.items():
        tags_str = ' '.join(map(str, volume_tags))
        f.write(f"{node_id}: {tags_str}\n")