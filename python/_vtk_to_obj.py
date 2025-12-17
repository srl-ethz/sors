import vtk
import meshio
import numpy as np
import os

def vtk_points_to_numpy(vtk_points):
    """Converts vtkPoints to a numpy array."""
    num_points = vtk_points.GetNumberOfPoints()
    points_np = np.empty((num_points, 3), dtype=np.float64)
    for i in range(num_points):
        points_np[i, :] = vtk_points.GetPoint(i)
    return points_np


def vtk_cells_to_numpy(vtk_cells):
    """Converts vtkCellArray to a numpy array of triangles."""
    triangles = []
    cell_array = vtk_cells
    cell_array.InitTraversal()
    id_list = vtk.vtkIdList()
    while cell_array.GetNextCell(id_list):
        num_ids = id_list.GetNumberOfIds()
        if num_ids == 3:
            triangles.append([id_list.GetId(0), id_list.GetId(1), id_list.GetId(2)])
    return np.array(triangles, dtype=np.uint32)


def convert_vtu_to_obj(vtu_path, obj_path):
    """Converts a .vtu file with tetrahedral cells to .obj format."""
    print(f"Converting {vtu_path} to {obj_path}")

    # Read the VTU file
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_path)
    reader.Update()

    # Extract the surface mesh from the tetrahedral mesh
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputConnection(reader.GetOutputPort())
    surface_filter.Update()

    # Convert VTK surface mesh to numpy arrays
    surface_mesh = surface_filter.GetOutput()
    points = surface_mesh.GetPoints()
    cells = surface_mesh.GetPolys()

    # Convert points and cells to numpy arrays
    points_np = vtk_points_to_numpy(points)
    cells_np = vtk_cells_to_numpy(cells)

    # Create meshio mesh
    mesh = meshio.Mesh(
        points=points_np,
        cells=[("triangle", cells_np)]
    )
    # Write the mesh to an OBJ file
    meshio.write(obj_path, mesh)


def convert_vtu_folder_to_obj(vtu_folder, obj_folder):
    """Converts all .vtu files in a folder to .obj format."""
    if not os.path.exists(obj_folder):
        os.makedirs(obj_folder)

    vtu_files = [f for f in os.listdir(vtu_folder) if f.endswith('.vtu')]

    for vtu_file in vtu_files:
        vtu_path = os.path.join(vtu_folder, vtu_file)
        obj_path = os.path.join(obj_folder, vtu_file.replace('.vtu', '.obj'))
        convert_vtu_to_obj(vtu_path, obj_path)

    print("Conversion complete.")


vtu_folder = "output_fusimuscle"
obj_folder = "output_fusimuscle_obj"

convert_vtu_folder_to_obj(vtu_folder, obj_folder)