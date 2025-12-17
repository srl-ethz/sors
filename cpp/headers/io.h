#ifndef IO_H
#define IO_H

#include "common.h"
#include "vtu11/vtu11.hpp"
#include <mshio/mshio.h>

#include "plane3D.h"

/**
 * @class IO
 * @brief Static utility collection for mesh and field input/output.
 *
 * VTU implementation based on https://github.com/phmkopp/vtu11
 * The IO class provides helper routines for:
 * - exporting tetrahedral / hexahedral meshes and fields to VTU/PVD (ParaView)
 * - reading tetrahedral / hexahedral meshes from Gmsh .msh files
 * - extracting surface patches and mapping them to local element faces
 * - handling simple multi-material tagging via per-vertex and per-element tags
 * - basic utilities such as appending vectors to a CSV-like log file
 */
class IO
{
public:
    IO() {};

    /**
     * @brief Writes a tetrahedral mesh and optional fields/planes to a VTU file.
     *
     * Exports:
     * - point coordinates from @p vertices
     * - tetrahedral connectivity from @p elements
     * - optional scalar/vector fields provided via @p visuals
     * - optional visualization quads generated from @p planes
     *
     * The file will be written as <filename>.vtu in VTK unstructured-grid format.
     *
     * @param filename Base filename (without extension) for the VTU file.
     * @param vertices N×3 matrix of vertex coordinates.
     * @param elements M×4 matrix of tetrahedral vertex indices.
     * @param visuals  Optional list of datasets (per-vertex or per-element) to attach.
     * @param planes   Optional list of Plane3D objects to visualize as quad cells.
     * @return true on success, false otherwise.
     */
    static bool save_tet_VTU(const std::string filename, 
                            Matrix<double, -1, 3>& vertices, 
                            Matrix<int, -1, 4>& elements, 
                            std::vector<VisualOption> visuals = {}, 
                            std::vector<Plane3D> = {});
    
    /**
     * @brief Writes a hexahedral mesh and optional fields/planes to a VTU file.
     *
     * Exports:
     * - point coordinates from @p vertices
     * - hexahedral connectivity from @p elements
     * - optional scalar/vector fields provided via @p visuals
     * - optional visualization quads generated from @p planes
     *
     * The file will be written as <filename>.vtu in VTK unstructured-grid format.
     *
     * @param filename Base filename (without extension) for the VTU file.
     * @param vertices N×3 matrix of vertex coordinates.
     * @param elements M×8 matrix of hexahedral vertex indices.
     * @param visuals  Optional list of datasets (per-vertex or per-element) to attach.
     * @param planes   Optional list of Plane3D objects to visualize as quad cells.
     * @return true on success, false otherwise.
     */
    static bool save_hex_VTU(const std::string filename, 
                                 Matrix<double, -1, 3>& vertices, 
                                 Matrix<int, -1, 8>& elements, 
                                 std::vector<VisualOption> visuals = {},
                                 std::vector<Plane3D> planes = {});

                        
    /**
     * @brief Writes a PVD collection file referencing a time series of VTU files.
     *
     * Generates a ParaView .pvd file that links files named
     * <filename>_0.vtu, <filename>_1.vtu, ..., <filename>_(timesteps-1).vtu,
     * each associated with a physical time value i * dt.
     *
     * @param filename     Base name of the PVD file (without extension).
     * @param outputFolder Directory in which the PVD file is written.
     * @param timesteps    Number of time steps / VTU files in the series.
     * @param dt           Time step size used for the "timestep" attribute.
     * @return true on success, false otherwise.
     */ 
    static bool save_pvd (const std::string filename,
                          const std::string outputFolder,
                          const int timesteps,
                          const double dt);

     /**
     * @brief Reads a tetrahedral mesh from a Gmsh .msh file.
     *
     * Parses the node and element blocks, builds a contiguous vertex array,
     * and extracts all tetrahedral elements (Gmsh element type 4). Non-tet
     * elements are ignored.
     *
     * @param filename    Path to the .msh file.
     * @param vertices    Output N×3 matrix of vertex coordinates.
     * @param elements    Output M×4 matrix of tetrahedral vertex indices.
     * @param verboseFlag If true, prints basic information about node/element blocks.
     * @return true on success, false otherwise.
     */                     
    static bool read_tet_MSH (const std::string filename,
                              Matrix<double, -1, 3>& vertices,
                              Matrix<int, -1, 4>& elements,
                              bool verboseFlag = false);

    /**
     * @brief Reads a hexahedral mesh from a Gmsh .msh file.
     *
     * Parses the node and element blocks, builds a contiguous vertex array,
     * and extracts all hexahedral elements (Gmsh element type 5). Non-hex
     * elements are ignored.
     *
     * @param filename    Path to the .msh file.
     * @param vertices    Output N×3 matrix of vertex coordinates.
     * @param elements    Output M×8 matrix of hexahedral vertex indices.
     * @param verboseFlag If true, prints basic information about node/element blocks.
     * @return true on success, false otherwise.
     */
    static bool read_hex_MSH (const std::string filename,
                              Matrix<double, -1, 3>& vertices,
                              Matrix<int, -1, 8>& elements,
                              bool verboseFlag = false);

    /**
     * @brief Extracts connected surface triangle groups from a tetrahedral mesh.
     *
     * Finds all boundary faces (triangles) of a tet mesh, orients them consistently
     * outward based on element centers, and groups faces into connected "surface
     * patches".
     *
     * @param vertices    N×3 matrix of vertex coordinates.
     * @param eleIdx      M×4 matrix of tetrahedral vertex indices.
     * @param verboseFlag If true, prints statistics about surface extraction.
     * @return Tuple:
     *         - surfaceGroups: [nGroups][nFaces][3] vertex indices for each surface triangle
     *         - faceEleGroups: [nGroups][nFaces] element index for each surface face
     */
    static std::tuple<std::vector<std::vector<std::vector<int>>>, std::vector<std::vector<int>>> 
    extract_tet_surfaces (Matrix<double, -1, 3>& vertices,
                          Matrix<int, -1, 4>& eleIdx,
                          bool verboseFlag = false);

    /**
     * @brief Extracts connected surface quad groups from a hexahedral mesh.
     *
     * Finds all boundary faces (quads) of a hex mesh, orients them consistently
     * outward based on element centers, and groups faces into connected "surface
     * patches".
     *
     * @param vertices    N×3 matrix of vertex coordinates.
     * @param eleIdx      M×8 matrix of hexahedral vertex indices.
     * @param verboseFlag If true, prints statistics about surface extraction.
     * @return Tuple:
     *         - surfaceGroups: [nGroups][nFaces][4] vertex indices for each surface quad
     *         - faceEleGroups: [nGroups][nFaces] element index for each surface face
     */
    static std::tuple<std::vector<std::vector<std::vector<int>>>, std::vector<std::vector<int>>> 
    extract_hex_surfaces (Matrix<double, -1, 3>& vertices,
                          Matrix<int, -1, 8>& eleIdx,
                          bool verboseFlag = false);


    /**
     * @brief Converts global surface faces to local element-face indices.
     *
     * Given global surface groups (vertex IDs) and the owning element indices,
     * this function computes, for each element, the local face-vertex indices
     * (0..3 for tets) and the associated group IDs.
     *
     * @param elements      M×4 matrix of tetrahedral vertex indices.
     * @param surfaceGroups Surface face groups as returned by extract_tet_surfaces().
     * @param faceEleGroups Element indices per face as returned by extract_tet_surfaces().
     * @return Tuple:
     *         - surfaceFaces: per-element list of local face-vertex indices (as doubles)
     *         - surfaceIdx:   per-element list of surface-group IDs (as doubles)
     */
    static std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> 
    global_to_local_surfaces (Matrix<int, -1, 4>& elements,
                              std::vector<std::vector<std::vector<int>>>& surfaceGroups,
                              std::vector<std::vector<int>>& faceEleGroups);

    /**
     * @brief Reads per-vertex material tags from a simple text file.
     *
     * The file format is one vertex per line:
     *   <vertexIndex>: <tag1> <tag2> ...
     *
     * This function returns a vector where entry i is the set of all tags
     * assigned to vertex i.
     *
     * @param vertexTagPath Path to the vertex tag file.
     * @return Vector of sets, vertexTags[i] contains all tags for vertex i.
     */
    static std::vector<std::set<int>> get_multimaterial_vertex_tags(const std::string& vertexTagPath);

    /**
     * @brief Derives per-element material tags from per-vertex tags.
     *
     * For each element, looks up the tag sets of its vertices and chooses
     * the tag with the highest frequency as the element tag (majority vote).
     *
     * @param vertexTags Vector of per-vertex tag sets as returned by get_multimaterial_vertex_tags().
     * @param elementIdx M×4 matrix of tetrahedral vertex indices.
     * @return Vector of size M containing the selected tag for each element.
     */
    static std::vector<int> get_multimaterial_element_tags(std::vector<std::set<int>> vertexTags,
                                                           Matrix<int, -1, 4>& elementIdx);

    /**
     * @brief Appends a VectorXd as a CSV-formatted row to a text file.
     *
     * Opens @p filePath in append mode and writes the entries of @p vector
     * as comma-separated values followed by a newline. Creates the file if
     * it does not exist.
     *
     * @param filePath Path to the output text/CSV file.
     * @param vector   Vector of values to append as one row.
     */
    static void appendRowToFile(const std::string& filePath,
                                const Eigen::VectorXd& vector);
    
    /**
     * @brief Converts each hexahedron into five tetrahedra using a fixed pattern.
     *
     * For every input hex (8 vertex indices), generates 5 tetrahedra that fill
     * the same volume. The decomposition assumes the standard VTK/Gmsh vertex
     * ordering:
     * NB: it works only if the vertices of the hexahedrons are numbered as follows:
     *        5 -------- 6
     *       /|         /|       Base face:‌ {0, 1, 2, 3}.
     *      4 -------- 7 |       Upper face {4, 5, 6, 7}.
     *      | |        | |       Vertical edges: {0-4}, {1-5}, {2-6}, {3-7}.
     *      | 1 -------|-2       This is the standard order of the hex vertices
     *      |/         |/        for vtk meshes or gmsh.
     *      0 -------- 3
     * 
     * 
     *
     * @param hex_elements M×8 matrix of hexahedral vertex indices.
     * @return (5M)×4 matrix of tetrahedral vertex indices.
     */
    static Eigen::Matrix<int, -1, 4> hex_to_5_tets (const Eigen::Matrix<int, -1, 8>& hex_elements);

    /**
     * @brief Constructs a regular hexahedral voxel grid.
     *
     * Builds a structured grid of size (nx × ny × nz) hexahedral cells with
     * voxel spacings (dx, dy, dz). Vertices are laid out on a Cartesian grid,
     * and elements reference these vertices in standard hex ordering.
     *
     * @param nx Number of cells in x-direction.
     * @param ny Number of cells in y-direction.
     * @param nz Number of cells in z-direction.
     * @param dx Cell size in x-direction.
     * @param dy Cell size in y-direction.
     * @param dz Cell size in z-direction.
     * @return Pair:
     *         - vertices: ( (nx+1)(ny+1)(nz+1) )×3 matrix of vertex coordinates
     *         - elements: (nx·ny·nz)×8 matrix of hexahedral vertex indices
     */
    static std::pair<Eigen::Matrix<double, -1, 3>, Eigen::Matrix<int, -1, 8>> make_voxel_hex_grid(int nx, int ny, int nz,
                                                                                                   double dx, double dy, double dz);
};

#endif
