### Utility functions
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import meshio
from scipy.spatial import cKDTree
from typing import Tuple, Dict, Any
from py_sors import IO


### Colors from https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def best_fit_transform (A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
    A: Nxm numpy array of corresponding points
    B: Nxm numpy array of corresponding points
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    '''
    # Get number of dimensions
    m = A.shape[1]

    # Translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # Rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1,:] *= -1
        R = np.dot(Vt.T, U.T)

    # Translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # Homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def icp (real_q, B):
    """
    Returns result from rigid registration on markers. A is moved to match B. Give a whole time sequence of markers for A of real data.
    """
    init_pose=None
    max_iterations=200
    tolerance=1e-5

    A = real_q[0]

    # Get number of dimensions
    m = A.shape[1]

    # Make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    aligned_q =  np.ones((real_q.shape[0], m+1, A.shape[0]))
    aligned_q[:,:m,:] = np.copy(np.transpose(real_q, (0,2,1)))
        
    # Apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    def nearest_neighbor(src, dst):
        '''
        Find the nearest (Euclidean) neighbor in dst for each point in src
        Input:
            src: Nxm array of points
            dst: Nxm array of points
        Output:
            distances: Euclidean distances of the nearest neighbor
            indices: dst indices of the nearest neighbor
        '''

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()

    for _ in range(max_iterations):
        # Find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # Compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # Update the current source
        src = np.dot(T, src)

        # Check error
        mean_error = np.mean(distances)
        print(f"Mean Error: {mean_error:.4e}")
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # Calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)
    aligned_q = np.einsum('ij, bjk -> bik', T, aligned_q)
    # breakpoint()
    return np.transpose(aligned_q, (0,2,1))[:,:,:3] #, T, distances, indices


def compute_marker_interpolation (vertices, elements, markers):
    """
    Compute per-marker tetra interpolation data (tet index + barycentric weights).
    For each marker, we:
      1) pick the closest tetrahedron by centroid distance,
      2) compute barycentric weights within that tetrahedron (w0..w3),
    so the marker position can later be reconstructed via:
        p = w0*v0 + w1*v1 + w2*v2 + w3*v3
    Args:
        vertices: (N, 3) vertex positions.
        elements: (M, 4) tetra vertex indices.
        markers:  (K, 3) marker positions.
    Returns:
        closestElements: list[int] of length K, tetra index per marker.
        interpolationCoeffs: list[np.ndarray] of length K, each (4,) barycentric weights.
    """

    # Find the closest element for each marker
    closestElements = [-1] * len(markers)
    closestDistances = np.inf * np.ones(len(markers))

    for i, ele in enumerate(elements):
        ele_vertices = vertices[ele]
        # Calculate the centroid of the element
        centroid = np.mean(ele_vertices, axis=0)
        # Calculate the distance from the centroid to each marker
        for j, marker in enumerate(markers):
            dist = np.linalg.norm(marker - centroid)
            if dist < closestDistances[j]:
                closestDistances[j] = dist
                closestElements[j] = i
    
    assert all(ce >= 0 for ce in closestElements), "Not all markers found a closest element."

    # Barycentric interpolation coefficients for tetrahedra
    interpolationCoeffs = []
    for i, ele_idx in enumerate(closestElements):
        v = vertices[elements[ele_idx]]
        T = np.stack([
            v[0] - v[3],
            v[1] - v[3],
            v[2] - v[3]
        ], axis=1)
        barycentricCoords = np.linalg.solve(T, markers[i] - v[3])
        barycentricCoords = np.append(barycentricCoords, 1 - np.sum(barycentricCoords))
        interpolationCoeffs.append(barycentricCoords)
        markerLoc = barycentricCoords[0] * v[0] + barycentricCoords[1] * v[1] + barycentricCoords[2] * v[2] + barycentricCoords[3] * v[3]
        assert np.allclose(markerLoc, markers[i]), f"Marker {i} location mismatch: {markerLoc} vs {markers[i]}"

    return closestElements, interpolationCoeffs


def marker_interpolation (vertices, elements, closestElements, interpolationCoeffs):
    # Interpolate the marker positions using the barycentric coordinates
    interpolatedMarkers = []
    for ele, coeff in zip(closestElements, interpolationCoeffs):
        v = vertices[elements[ele]]
        interpolatedMarkers.append(coeff[0] * v[0] + coeff[1] * v[1] + coeff[2] * v[2] + coeff[3] * v[3])
    return np.stack(interpolatedMarkers, axis=0)


def _tri_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute the area of each triangle in a surface mesh.
    Args:
        vertices (np.ndarray): Shape (N, 3), Vertex coordinates (x, y, z).
        faces (np.ndarray): Shape (K, 3), Array of triangle definitions, each row is 3 vertex indices into `vertices`.
    Returns:
        np.ndarray: Shape (K,) Area of each triangle in the same order as `faces`.
    """
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def _sample_on_triangles(vertices: np.ndarray,
                         faces: np.ndarray,
                         n_samples: int,
                         seed: int | None = None) -> np.ndarray:
    """Uniformly sample points on a triangle mesh (weighted by triangle area).
    Args:
        vertices (np.ndarray): Shape (N, 3), Vertex coordinates (x, y, z).
        faces (np.ndarray): Shape (K, 3), Array of triangle definitions, each row is 3 vertex indices into `vertices`.
        n_samples (int): Number of points to sample.
        seed (int or None): Random seed for reproducibility. If None, RNG is not seeded.
    Returns:
        np.ndarray: Shape (n_samples, 3): Sampled 3D points uniformly distributed over the mesh surface.
    """
    rng = np.random.default_rng(seed)
    # Triangle areas for the current (possibly deformed) mesh surface
    areas = _tri_areas(vertices, faces) # shape: (K,), K = number of surface triangles
    total = areas.sum()
    if total <= 0:
        # Degenerate surface; just return random vertices
        idx = rng.integers(0, vertices.shape[0], size=n_samples)
        return vertices[idx]

    # Build a discrete distribution over triangles, proportional to area
    pdf = areas / total  # probability density function
    cdf = np.cumsum(pdf) # cumulative distribution function
    
    # Inverse transform sampling: pick which triangle each point comes from
    r = rng.random(n_samples)
    tri_ids = np.searchsorted(cdf, r, side='right') # shape: (n_samples,)

    # Uniformly sample inside each chosen triangle (barycentric “sqrt trick”)
    r1 = rng.random(n_samples) # u+v+w=1, u,v,w>=0
    r2 = rng.random(n_samples)
    # u+v+w=1, u,v,w>=0
    u = 1.0 - np.sqrt(r1)
    v = np.sqrt(r1) * (1.0 - r2)
    w = np.sqrt(r1) * r2

    # Map barycentric coords to 3D points
    a = vertices[faces[tri_ids, 0]] # shape: (n_samples, 3)
    b = vertices[faces[tri_ids, 1]]
    c = vertices[faces[tri_ids, 2]]
    pts = (u[:, None] * a) + (v[:, None] * b) + (w[:, None] * c)
    return pts  # shape: (n_samples, 3)


def _read_and_transform_mesh(meshFilePath: str,
               scale: float = 1.0,
               centerX: bool = False,
               centerY: bool = False,
               centerZ: bool = False,
               translationX: int = 0, 
               translationY: int = 0, 
               translationZ: int = 0
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """
    Read a tetrahedral mesh or triangle-surface mesh from file (readable by meshio), optionally transform, order: 1) scale, (2) center, (3) translate.
    Args:
        meshFilePath (str): Path to the mesh file readable by meshio (.msh, .vtu, ...).
        scale (float): Scaling factor to apply to the mesh.
        centerX (bool): If True, shift mesh so its X-center is at 0.
        centerY (bool): If True, shift mesh so its Y-center is at 0.
        centerZ (bool): If True, shift mesh so its Z-center is at 0.
        translationX (int): If non-zero, translate mesh along X by this amount.
        translationY (int): If non-zero, translate mesh along Y by this amount.
        translationZ (int): If non-zero, translate mesh along Z by this amount.
    Returns:
        vertices (np.ndarray): shape (N, 3), float: Vertex coordinates 
        elements (np.ndarray): shape (M, 4) for tetra meshes, (K,3) for surface meshes
        transformInfo (tuple[float,float,float,float]): (scale, tx, ty, tz) applied to the mesh.
    Raises:
        AssertionError: If the mesh format is unsupported or invalid.
    """
    # Get vertices and elements
    mesh = meshio.read(meshFilePath)
    vertices = mesh.points
    elements = None
    for cellBlock in mesh.cells:
        if cellBlock.type == "tetra":
            elements = cellBlock.data # If tetra mesh, use tetra elements only
            break
        if cellBlock.type == "triangle": # If surface mesh, use triangle elements only
            elements = cellBlock.data
            break
    assert elements is not None, "No tetrahedral or triangle elements found in mesh file."
    
    # Track transforms
    tx = ty = tz = 0.0
    applied_scale = 1.0
    
    # 1) Scaling
    if scale is not None:
        vertices *= scale
        applied_scale = scale
        
    # 2) Centering
    if centerX:
        xmin, xmax = np.min(vertices[:, 0]), np.max(vertices[:, 0])
        shift = (xmin + xmax) / 2.0
        vertices[:, 0] -= shift
        tx = shift
    if centerY:
        ymin, ymax = np.min(vertices[:, 1]), np.max(vertices[:, 1])
        shift = (ymin + ymax) / 2.0
        vertices[:, 1] -= shift
        ty = shift
    if centerZ:
        zmin, zmax = np.min(vertices[:, 2]), np.max(vertices[:, 2])
        shift = (zmin + zmax) / 2.0
        vertices[:, 2] -= shift
        tz = shift
        
    # 3) Translation
    if translationX != 0:
        vertices[:, 0] -= translationX
        tx += translationX
    if translationY != 0:
        vertices[:, 1] -= translationY
        ty += translationY
    if translationZ != 0:
        vertices[:, 2] -= translationZ
        tz += translationZ
        
    # transform info
    transformInfo = (applied_scale, tx, ty, tz)
    
    return vertices, elements, transformInfo


def _chamfer_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    Symmetric Chamfer distance between two point clouds A and B.
    A: (Na,3), B: (Nb,3)
    Returns: sqrt (mean_a min_b ||a-b||^2 + mean_b min_a ||b-a||^2)  (units: length)
    """
    treeA = cKDTree(A)
    treeB = cKDTree(B)
    dAB, _ = treeB.query(A, k=1, workers=-1) # k = 1: return th nearest neighbour, workers=-1: use all available cores
    dBA, _ = treeA.query(B, k=1, workers=-1)
    chamfer_distance_squared = float(np.mean(dAB**2) + np.mean(dBA**2))
    chamfer_distance = np.sqrt(chamfer_distance_squared)
    return chamfer_distance


def _make_disk_mesh(center: np.ndarray,
                    normal: np.ndarray,
                    radius: float,
                    n_segments: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a flat disk triangulated as a fan.
    Returns:
        V: (n_segments+1, 3) vertices, first is center
        F: (n_segments, 3) triangle indices
    """
    c = np.asarray(center, float)
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) + 1e-12)

    # Build orthonormal basis (u,v,n)
    # Pick a vector not parallel to n
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a); u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)

    # Vertices: center + ring
    V = np.empty((n_segments + 1, 3), dtype=float)
    V[0] = c
    angles = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
    ring = c + radius * (np.cos(angles)[:, None] * u[None, :] + np.sin(angles)[:, None] * v[None, :])
    V[1:] = ring

    # Faces: fan around the center
    F = np.empty((n_segments, 3), dtype=int)
    for i in range(n_segments):
        j = 1 + i
        k = 1 + ((i + 1) % n_segments)
        # Winding so the normal points ~ +n (flip if you want the other side)
        F[i] = [0, j, k]

    return V, F
    
    
################################################################################
# Utilities to compute geometric and topological statistics for a tetrahedral mesh.
# This script reads a tetrahedral mesh from a file, computes various statistics
# such as volume, surface area, edge lengths, and saves the results to a JSON file

def _tetra_volumes(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Compute volumes of tetrahedra from vertex coordinates.
    Args:
        V: (N,3) float array of vertex positions.
        T: (M,4) int array of tetrahedron vertex indices.
    Returns:
        (M,) array of absolute volumes for each tet.
    """
    v0 = V[T[:,0]]; v1 = V[T[:,1]]; v2 = V[T[:,2]]; v3 = V[T[:,3]]
    M = np.cross(v1 - v0, v2 - v0)
    vol = np.einsum('ij,ij->i', M, v3 - v0) / 6.0
    return np.abs(vol)


def _edge_length_stats(V: np.ndarray, T: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics over all unique tetrahedral edges.
    Args:
        V: (N,3) float array of vertex positions.
        T: (M,4) int array of tetrahedron vertex indices.
    Returns:
        Dictionary with min, max, mean, std of edge lengths and count of unique edges.
    """
    # Collect unique edges into a set of sorted tuples
    edgeSet = set()
    for tet in T:
        i0, i1, i2, i3 = tet
        edges = [(i0, i1), (i0, i2), (i0, i3), (i1, i2), (i1, i3), (i2, i3)]
        for e in edges:
            edgeSet.add(tuple(sorted(e)))
    edges = np.array(list(edgeSet), dtype=int)
    d = V[edges[:,0]] - V[edges[:,1]]
    lens = np.linalg.norm(d, axis=1)
    return {
        "edge_min": float(lens.min() if len(lens) else 0.0),
        "edge_max": float(lens.max() if len(lens) else 0.0),
        "edge_mean": float(lens.mean() if len(lens) else 0.0),
        "edge_std": float(lens.std(ddof=0) if len(lens) else 0.0),
        "n_edges": int(len(lens))
    }


def _compute_stats(V: np.ndarray, T: np.ndarray) -> Dict[str, Any]:
    """
    Compute all mesh statistics from vertices and tets.
    Args:
        V: (N,3) float array of vertex positions.
        T: (M,4) int array of tetrahedron vertex indices.
    Returns:
        Dictionary with:
          - vertex/tet counts
          - surface tri/vertex counts
          - volume, surface area
          - bounding box min/max/diag, centroid
          - edge length stats
    """
    surfaceGroups, _ = IO.extract_tet_surfaces(V, T)
    surfaceFaces = np.vstack(surfaceGroups)  

    vol = float(np.sum(_tetra_volumes(V, T)))
    area = float(np.sum(_tri_areas(V, surfaceFaces)))
    bbox_min = V.min(axis=0).tolist()
    bbox_max = V.max(axis=0).tolist()
    bbox_diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    centroid = V.mean(axis=0).tolist()
    edge_stats = _edge_length_stats(V, T)
    surf_vertices = sorted(set(surfaceFaces.flatten().tolist()))
    stats = {
        "n_vertices": int(V.shape[0]),
        "n_tets": int(T.shape[0]),
        "n_surface_tris": int(surfaceFaces.shape[0]),
        "n_surface_vertices": int(len(surf_vertices)),
        "volume_m3": vol,
        "surface_area_m2": area,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "bbox_diag": bbox_diag,
        "centroid": centroid,
    }
    stats.update(edge_stats)
    return stats
