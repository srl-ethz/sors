#include "io.h"

bool IO::save_tet_VTU (const std::string filename, Matrix<double, -1, 3>& vertices, Matrix<int, -1, 4>& elements, std::vector<VisualOption> visuals, std::vector<Plane3D> planes) {
    // Store the mesh in a .vtu file

    // STEP 1: First create mesh data for tetrahedral elements
    // Create point data
    int numVerticesTet = vertices.rows(); 
    VectorXd reshapedVertices = vertices.reshaped<RowMajor>(); // Reshape needed for RowMajor ordering
    std::vector<double> points(reshapedVertices.data(), reshapedVertices.data()+reshapedVertices.size());
    // Create connectivity data
    int numElementsTet = elements.rows(); 
    VectorXi reshapedElements = elements.reshaped<RowMajor>(); // Reshape needed for RowMajor ordering
    std::vector<vtu11::VtkIndexType> connectivity(reshapedElements.data(), reshapedElements.data()+reshapedElements.size());
    // Create offsets data
    std::vector<vtu11::VtkIndexType> offsets(numElementsTet);
    for (int i = 0; i < numElementsTet; i++) {
        offsets[i] = 4 * (i + 1);
    }
    // Create cell type data of each cell, Triangle 5, Quads are 9, Tetrahedra are 10, Hexahedron 12
    std::vector<vtu11::VtkCellType> types(numElementsTet, 10);

    // STEP 2: add plane data to the mesh
    int numPlanes = planes.size();
    int numVerticesAll = numVerticesTet;
    int numElementsAll = numElementsTet;
    if (!planes.empty()) {
        std::vector<vtu11::VtkIndexType> planeConnectivity = {0, 1, 2, 3}; 
        for (auto& plane : planes) {
            // Visualize planes
            std::vector<double> planePoints = plane.get_square_points();
            // Update points
            points.insert(points.end(), planePoints.begin(), planePoints.end());
            // Update connectivity
            int planeOffset = numVerticesAll; // Offset the plane's connectivity by the number of vertices in the mesh
            for (auto& index : planeConnectivity) {
                connectivity.push_back(index + planeOffset);
            }
            // Update offsets
            offsets.push_back(4 * (numElementsAll + 1)); // Offset for the plane (which has 4 vertices)
            // Update types
            types.push_back(9); // Add the plane cell type (quad is 9)
            // Update number of vertices and elements
            numVerticesAll = numVerticesAll + 4;
            numElementsAll = numElementsAll + 1;
        }
    }
    vtu11::Vtu11UnstructuredMesh mesh {points, connectivity, offsets, types};

    // STEP 3: Create additional data for the mesh
    // Default datasets we always add: Displacement, defaultElementColor

    // Displacement: scalar per vertex (x+y+z)
    VisualOption displacement;
    displacement.name = "Displacement";
    displacement.type = VisualType::ScalarVertex;
    displacement.data = MatrixXd::Zero(1, numVerticesTet);
    displacement.threshold = -std::numeric_limits<double>::infinity(); // No threshold
    for (int i = 0; i < numVerticesTet; i++) {
        displacement.data(0,i) = vertices(i, 0) + vertices(i, 1) + vertices(i, 2);
    }
    visuals.push_back(displacement);
    // Default element color: 1 for hex cells, 0 for planes (via padding)
    VisualOption defaultElementColor;
    defaultElementColor.name = "defaultElementColor";
    defaultElementColor.type = VisualType::ScalarElement;
    defaultElementColor.data = MatrixXd::Ones(1, numElementsTet);
    defaultElementColor.threshold = -std::numeric_limits<double>::infinity(); // no threshold
    visuals.push_back(defaultElementColor);
    
    // Contains tuples of (name, dataset type, number of components)
    // Dataset types: vtu11::DataSetType::PointData = per vertices, vtu11::DataSetType::CellData = per element
    // Number of components: 1 for scalar, 3 for vector
    std::vector<vtu11::DataSetInfo> dataSetInfo; 
    // Contains the actual data for each dataset 
    std::vector<std::vector<double>> DataSet;

    for (const auto& vo : visuals) {
        // Vertex-scalar data
        if(vo.type == VisualType::ScalarVertex) {
            if (vo.data.rows() != 1 || vo.data.cols() != numVerticesTet) {
                throw std::runtime_error(vo.name + " (ScalarVertex) must be 1 x numVerticesTet");
            } 
            dataSetInfo.push_back({vo.name, vtu11::DataSetType::PointData, 1});
            std::vector<double> info;
            info.reserve(numVerticesAll);
            for(int i = 0; i< numVerticesTet; i++) {
                if(vo.threshold < vo.data(0,i)) { 
                    info.push_back(vo.data(0,i));
                } else {
                    info.push_back(0.0);
                }
            }
            if (!planes.empty()) {
                    info.insert(info.end(), 4 * numPlanes, 0.0); // Add zeros for plane vertices (4 vertices)
            }
            DataSet.push_back(std::move(info));
        // Element-scalar data
        } else if(vo.type == VisualType::ScalarElement) {
            if (vo.data.rows() != 1 || vo.data.cols() != numElementsTet) {
                throw std::runtime_error(vo.name + " (ScalarElement) must be 1 x numElementsTet");
            } 
            dataSetInfo.push_back({vo.name, vtu11::DataSetType::CellData, 1 });
            std::vector<double> info;
            info.reserve(numElementsAll);
            for(int i = 0; i< numElementsTet; i++) {
                if(vo.threshold < vo.data(0,i)) { 
                    info.push_back(vo.data(0,i));
                } else {
                    info.push_back(0.0);
                }
            }
            if (!planes.empty()) {
                    info.insert(info.end(), numPlanes, 0.0);  // Add zeros for plane elements (1 component)
            }
            DataSet.push_back(std::move(info));
        // Vertex-vector data
        } else if(vo.type == VisualType::VectorVertex) {
            if (vo.data.rows() != 3 || vo.data.cols() != numVerticesTet) {
                throw std::runtime_error(vo.name + " (VectorVertex) must be 3 x numVerticesTet");
            } 
            dataSetInfo.push_back({vo.name, vtu11::DataSetType::PointData, 3 });
            std::vector<double> info;
            info.reserve(3 * numVerticesAll);
            for(int i = 0; i< numVerticesTet; i++) {
                double x = vo.data(0, i);
                double y = vo.data(1, i);
                double z = vo.data(2, i);
                double norm = std::sqrt(x * x + y * y + z * z);
                if (vo.threshold < norm) {
                    info.push_back(x);
                    info.push_back(y);
                    info.push_back(z);
                } else {
                    info.push_back(0.0);
                    info.push_back(0.0);
                    info.push_back(0.0);
                }
            }
            if (!planes.empty()) {
                    info.insert(info.end(), 12 * numPlanes, 0.0); // Add zeros for plane vertices (3 components x 4 vertices)
            }
            DataSet.push_back(std::move(info));
        // Element-vector data
        } else if(vo.type == VisualType::VectorElement) {
            if (vo.data.rows() != 3 || vo.data.cols() != numElementsTet) {
                throw std::runtime_error(vo.name + " (VectorElement) must be 3 x numElementsTet");
            } 
            dataSetInfo.push_back({vo.name, vtu11::DataSetType::CellData, 3 });
            std::vector<double> info;
            info.reserve(3 * numElementsAll);
            for(int i = 0; i< numElementsTet; i++) {
                double x = vo.data(0, i);
                double y = vo.data(1, i);
                double z = vo.data(2, i);
                double norm = std::sqrt(x * x + y * y + z * z);
                if (vo.threshold < norm) {
                    info.push_back(x);
                    info.push_back(y);
                    info.push_back(z);
                } else {
                    info.push_back(0.0);
                    info.push_back(0.0);
                    info.push_back(0.0);
                }
            }
            if (!planes.empty()) {
                    info.insert(info.end(), 3 * numPlanes, 0.0); // Add zeros for plane elements (3 components)
            } 
            DataSet.push_back(std::move(info));
        } else {
            throw std::runtime_error("Unknown VisualType");
        }
    }
    // STEP 4: Write VTU
    vtu11::writeVtu(filename+".vtu", mesh, dataSetInfo, DataSet, "Ascii");
    return true;
}
 

bool IO::save_hex_VTU(const std::string filename, Matrix<double, -1, 3>& vertices, Matrix<int, -1, 8>& elements, std::vector<VisualOption> visuals, std::vector<Plane3D> planes){
    // Store the mesh in a .vtu file
    
    // STEP 1: Create mesh data for hexahedral elements
    // Create point data
    int numVerticesHex = vertices.rows();
    VectorXd reshapedVertices = vertices.reshaped<RowMajor>(); // Row-major flatten
    std::vector<double> points(reshapedVertices.data(), reshapedVertices.data() + reshapedVertices.size());
    // Create connectivity data
    int numElementsHex = elements.rows();
    VectorXi reshapedElements = elements.reshaped<RowMajor>();
    std::vector<vtu11::VtkIndexType> connectivity(reshapedElements.data(), reshapedElements.data() + reshapedElements.size());
    // Offsets: cumulative vertex count per cell
    std::vector<vtu11::VtkIndexType> offsets(numElementsHex);
    for (int i = 0; i < numElementsHex; ++i) {
        offsets[i] = 8 * (i + 1); // 8 vertices per hex
    }
    // Cell types: 12 = VTK_HEXAHEDRON
    std::vector<vtu11::VtkCellType> types(numElementsHex, 12);

    // STEP 2: add plane data to the mesh (as quads)
    int numPlanes = planes.size();
    int numVerticesAll = numVerticesHex;
    int numElementsAll = numElementsHex;
    if (!planes.empty()) {
        int meshCumulativeOffset = offsets.empty() ? 0 : offsets.back();
        std::vector<vtu11::VtkIndexType> planeConnectivity = {0, 1, 2, 3}; 
        for (auto& plane : planes) {
            // Visualize planes
            std::vector<double> planePoints = plane.get_square_points();
            // Update points
            points.insert(points.end(), planePoints.begin(), planePoints.end());
            // Update connectivity
            int planeOffset = numVerticesAll; // Offset the plane's connectivity by the number of vertices in the mesh
            for (auto& index : planeConnectivity) {
                connectivity.push_back(index + planeOffset);
            }
            // Update offsets
            meshCumulativeOffset += 4; // 4 vertices per plane
            offsets.push_back(meshCumulativeOffset); // Offset for the plane (which has 4 vertices)
            // Update types
            types.push_back(9); // Add the plane cell type (quad is 9)
            // Update number of vertices and elements
            numVerticesAll = numVerticesAll + 4;
            numElementsAll = numElementsAll + 1;
        }
    }
    vtu11::Vtu11UnstructuredMesh mesh {points, connectivity, offsets, types};

    // STEP 3: Create additional data for the mesh
    // Default datasets we always add: Displacement, defaultElementColor

    // Displacement: scalar per vertex (x+y+z)
    VisualOption displacement;
    displacement.name = "Displacement";
    displacement.type = VisualType::ScalarVertex;
    displacement.data = MatrixXd::Zero(1, numVerticesHex);
    displacement.threshold = -std::numeric_limits<double>::infinity(); // No threshold
    for (int i = 0; i < numVerticesHex; ++i) {
        displacement.data(0, i) = vertices(i, 0) + vertices(i, 1) + vertices(i, 2);
    }
    visuals.push_back(displacement);
    // Default element color: 1 for hex cells, 0 for planes (via padding)
    VisualOption defaultElementColor;
    defaultElementColor.name = "defaultElementColor";
    defaultElementColor.type = VisualType::ScalarElement;
    defaultElementColor.data = MatrixXd::Ones(1, numElementsHex);
    defaultElementColor.threshold = -std::numeric_limits<double>::infinity();
    visuals.push_back(defaultElementColor);

    // DataSetInfo + DataSet (same pattern as tets)
    std::vector<vtu11::DataSetInfo> dataSetInfo;
    std::vector<std::vector<double>> DataSet;

    for (const auto& vo : visuals) {
        // Vertex-scalar data
        if (vo.type == VisualType::ScalarVertex) {
            if (vo.data.rows() != 1 || vo.data.cols() != numVerticesHex) {
                throw std::runtime_error(vo.name + " (ScalarVertex) must be 1 x numVerticesHex");
            }
            dataSetInfo.push_back({vo.name, vtu11::DataSetType::PointData, 1});
            std::vector<double> info;
            info.reserve(numVerticesAll);
            for (int i = 0; i < numVerticesHex; ++i) {
                if (vo.threshold < vo.data(0, i)) { 
                    info.push_back(vo.data(0, i));
                } else {
                    info.push_back(0.0);
                }
            }
            if (!planes.empty()) {
                info.insert(info.end(), 4 * numPlanes, 0.0);
            }
            DataSet.push_back(std::move(info));
        // Element-scalar data
        } else if (vo.type == VisualType::ScalarElement) {
            if (vo.data.rows() != 1 || vo.data.cols() != numElementsHex) {
                throw std::runtime_error(vo.name + " (ScalarElement) must be 1 x numElementsHex");
            }
            dataSetInfo.push_back({vo.name, vtu11::DataSetType::CellData, 1});
            std::vector<double> info;
            info.reserve(numElementsAll);
            for (int i = 0; i < numElementsHex; ++i) {
                if (vo.threshold < vo.data(0, i)) {
                    info.push_back(vo.data(0, i));
                } else {
                    info.push_back(0.0);
                }
            }
            if (!planes.empty()) {
                info.insert(info.end(), numPlanes, 0.0); 
            }
            DataSet.push_back(std::move(info));
        // Vertex-vector data
        } else if (vo.type == VisualType::VectorVertex) {
            if (vo.data.rows() != 3 || vo.data.cols() != numVerticesHex) {
                throw std::runtime_error(vo.name + " (VectorVertex) must be 3 x numVerticesHex");
            }
            dataSetInfo.push_back({vo.name, vtu11::DataSetType::PointData, 3});
            std::vector<double> info;
            info.reserve(3 * numVerticesAll);
            for (int i = 0; i < numVerticesHex; ++i) {
                double x = vo.data(0, i);
                double y = vo.data(1, i);
                double z = vo.data(2, i);
                double norm = std::sqrt(x * x + y * y + z * z);
                if (vo.threshold < norm) {
                    info.push_back(x);
                    info.push_back(y);
                    info.push_back(z);
                } else {
                    info.push_back(0.0);
                    info.push_back(0.0);
                    info.push_back(0.0);
                }
            }
            if (!planes.empty()) {
                info.insert(info.end(), 12 * numPlanes, 0.0); 
            }
            DataSet.push_back(std::move(info));
        // Element-vector data
        } else if (vo.type == VisualType::VectorElement) {
            if (vo.data.rows() != 3 || vo.data.cols() != numElementsHex) {
                throw std::runtime_error(
                    vo.name + " (VectorElement) must be 3 x numElementsHex");
            }
            dataSetInfo.push_back({vo.name, vtu11::DataSetType::CellData, 3});
            std::vector<double> info;
            info.reserve(3 * numElementsAll);
            for (int i = 0; i < numElementsHex; ++i) {
                double x = vo.data(0, i);
                double y = vo.data(1, i);
                double z = vo.data(2, i);
                double norm = std::sqrt(x * x + y * y + z * z);
                if (vo.threshold < norm) {
                    info.push_back(x);
                    info.push_back(y);
                    info.push_back(z);
                } else {
                    info.push_back(0.0);
                    info.push_back(0.0);
                    info.push_back(0.0);
                }
            }
            if (!planes.empty()) {
                info.insert(info.end(), 3 * numPlanes, 0.0);
            }
            DataSet.push_back(std::move(info));

        } else {
            throw std::runtime_error("Unknown VisualType");
        }
    }
    // STEP 4: Write VTU
    vtu11::writeVtu(filename + ".vtu", mesh, dataSetInfo, DataSet, "Ascii");
    return true;
}
    

bool IO::save_pvd (const std::string filename, const std::string outputFolder, const int timesteps=1, const double dt=1) {
    // Link all the .vtu files in a .pvd file 
    std::ofstream f;
    f.open(outputFolder + "/" + filename + ".pvd");
    f << "<?xml version='1.0' encoding='UTF-8'?>\n";
    f << "<VTKFile type='Collection' version='0.1' byte_order='LittleEndian' compressor='vtkZLibDataCompressor'>\n";
    f << "\t<Collection>\n";

    for (int i = 0; i < timesteps; i++) {
        f << "\t\t<DataSet timestep='" << i*dt << "' group='' part='0' file='" << filename << "_" << i << ".vtu'/>\n";
    }

    f << "\t</Collection>\n";
    f << "</VTKFile>";
    f.close();

    return true;
}


bool IO::read_tet_MSH (const std::string filename, Matrix<double, -1, 3>& vertices, Matrix<int, -1, 4>& elements, bool verboseFlag) {
    mshio::MshSpec spec = mshio::load_msh(filename);

    // --- VERTICES --- //
    auto& nodes = spec.nodes;

    std::size_t numNodeBlocks = nodes.num_entity_blocks;
    if (numNodeBlocks != 1 && verboseFlag) {
        std::cout << bcolors.WARNING << "Multiple node blocks (" << numNodeBlocks << ") detected in mesh, only first block of 3D vertices is used..." << bcolors.ENDC << std::endl;
    }

    // Store all blocks of 3D vertices.
    std::size_t numNodes = nodes.num_nodes;
    std::size_t dim = 3;

    vertices.resize(numNodes, dim);
    vertices.setZero();

    // Create map from node tag to vertex index
    std::map<int, int> nodeTagToIdx;

    // TODO: Figure out when we want to skip nodes. With nodemap, we can simply skip the for loop.
    std::size_t nodeIdx = 0;
    for (std::size_t i = 0; i < numNodeBlocks; i++) {
        auto& block = nodes.entity_blocks[i];
        // For now we don't handle parametric cases.
        assert(block.parametric == 0);
        if (verboseFlag) {
            std::cout << bcolors.OKBLUE << "Node Block " << i << " has " << block.num_nodes_in_block << " entries with dimension " << block.entity_dim << bcolors.ENDC << std::endl;
        }
        for (std::size_t j = 0; j < block.num_nodes_in_block; j++) {
            for (std::size_t k = 0; k < dim; k++) {
                vertices(nodeIdx+j, k) = block.data[j*dim + k];
            }
            // Update map
            nodeTagToIdx[block.tags[j]] = nodeIdx+j;
        }
        // Prepare next block of 3D vertices
        nodeIdx += block.num_nodes_in_block;
    }

    // --- ELEMENTS --- //
    auto& eles = spec.elements;

    // Only consider tetrahedra
    std::size_t numEleNodes = 4;
    std::size_t numEleBlocks = eles.num_entity_blocks;
    if (numEleBlocks != 1 && verboseFlag) {
        std::cout << bcolors.WARNING << "Multiple element blocks (" << numEleBlocks << ") detected in mesh, only first block of tetrahedrons is used..." << bcolors.ENDC << std::endl;
    }
    // Store all blocks of tetrahedra
    std::size_t numElements = 0;
    // Figure out total number of tetrahedra, TODO: Unsure what this entity_dim is, I thought 2D vs 3D, but that's incorrect.
    for (std::size_t i = 0; i < numEleBlocks; i++) {
        auto& block = eles.entity_blocks[i];
        if (block.element_type != 4) {
            if (verboseFlag) {
                std::cout << bcolors.WARNING << "Element Block " << i << " is of element type " << block.element_type << ", skipping..." << bcolors.ENDC << std::endl;
            }
            continue;
        }
        // Add to total number of elements.
        numElements += block.num_elements_in_block;
    }
    elements.resize(numElements, numEleNodes);
    elements.setZero();

    // Each element block is of one element type, we only want to consider tetrahedra of type 4.
    std::size_t eleIdx = 0;
    for (std::size_t i = 0; i < numEleBlocks; i++) {
        auto& block = eles.entity_blocks[i];
        if (block.element_type != 4) {
            continue;
        }
        if (verboseFlag) {
            std::cout << bcolors.OKBLUE << "Element Block " << i << " has " << block.num_elements_in_block << " entries with " << block.element_type << " nodes each of dimension " << block.entity_dim << bcolors.ENDC << std::endl;
        }
        for (std::size_t j = 0; j < block.num_elements_in_block; j++) {
            for (std::size_t k = 0; k < numEleNodes; k++) {
                // Subtract 1 because MSH is 1-indexed, and the first entry is the element index [etag ntag1 ntag2 ntag3 ntag4]
                // Map node tags to vertex indices
                int vertexIdx = nodeTagToIdx[block.data[j*(numEleNodes+1) + k + 1]];
                elements(eleIdx+j, k) = vertexIdx;
            }
        }
        // Prepare next block of tetrahedra
        eleIdx += block.num_elements_in_block;
    }
    // TODO: Mesh with multiple subcomponents.
    return true;
}


bool IO::read_hex_MSH (const std::string filename, Matrix<double, -1, 3>& vertices, Matrix<int, -1, 8>& elements, bool verboseFlag) {
    mshio::MshSpec spec = mshio::load_msh(filename);

    // --- VERTICES --- //
    auto& nodes = spec.nodes;

    std::size_t numNodeBlocks = nodes.num_entity_blocks;
    if (numNodeBlocks != 1 && verboseFlag) {
        std::cout << bcolors.WARNING << "Multiple node blocks (" << numNodeBlocks << ") detected in mesh, only first block of 3D vertices is used..." << bcolors.ENDC << std::endl;
    }

    // Store all blocks of 3D vertices.
    std::size_t numNodes = nodes.num_nodes;
    std::size_t dim = 3;
    vertices.resize(numNodes, dim);
    vertices.setZero();

    // Create map from node tag to vertex index
    std::map<int, int> nodeTagToIdx;

    // TODO: Figure out when we want to skip nodes. With nodemap, we can simply skip the for loop.
    std::size_t nodeIdx = 0;
    for (std::size_t i = 0; i < numNodeBlocks; i++) {
        auto& block = nodes.entity_blocks[i];

        // For now we don't handle parametric cases.
        assert(block.parametric == 0);

        if (verboseFlag) {
            std::cout << bcolors.OKBLUE << "Node Block " << i << " has " << block.num_nodes_in_block << " entries with dimension " << block.entity_dim << bcolors.ENDC << std::endl;
        }
        for (std::size_t j = 0; j < block.num_nodes_in_block; j++) {
            for (std::size_t k = 0; k < dim; k++) {
                vertices(nodeIdx+j, k) = block.data[j*dim + k];
            }
            // Update map
            nodeTagToIdx[block.tags[j]] = nodeIdx+j;
        }
        // Prepare next block of 3D vertices
        nodeIdx += block.num_nodes_in_block;
    }

    // --- ELEMENTS --- //
    auto& eles = spec.elements;

    // Only consider hexaedra
    std::size_t numEleNodes = 8;
    std::size_t numEleBlocks = eles.num_entity_blocks;
    if (numEleBlocks != 1 && verboseFlag) {
        std::cout << bcolors.WARNING << "Multiple element blocks (" << numEleBlocks << ") detected in mesh, only first block of hexahedrons is used..." << bcolors.ENDC << std::endl;
    }

    // Store all blocks of hexaedra
    std::size_t numElements = 0;
    // Figure out total number of hexaedra, TODO: Unsure what this entity_dim is, I thought 2D vs 3D, but that's incorrect.
    for (std::size_t i = 0; i < numEleBlocks; i++) {
        auto& block = eles.entity_blocks[i];
        // element_type = 5 for hex: see Mshio documentation
        if (block.element_type != 5) {
            if (verboseFlag) {
                std::cout << bcolors.WARNING << "Element Block " << i << " is of element type " << block.element_type << ", skipping..." << bcolors.ENDC << std::endl;
            }
            continue;
        }
        // Add to total number of elements.
        numElements += block.num_elements_in_block;
    }
    elements.resize(numElements, numEleNodes);
    elements.setZero();

    // Each element block is of one element type, we only want to consider hexaedra of type 5.
    std::size_t eleIdx = 0;
    for (std::size_t i = 0; i < numEleBlocks; i++) {
        auto& block = eles.entity_blocks[i];
        if (block.element_type != 5) {
            continue;
        }
        if (verboseFlag) {
            std::cout << bcolors.OKBLUE << "Element Block " << i << " has " << block.num_elements_in_block << " entries with " << block.element_type << " nodes each of dimension " << block.entity_dim << bcolors.ENDC << std::endl;
        }

        for (std::size_t j = 0; j < block.num_elements_in_block; j++) {
            for (std::size_t k = 0; k < numEleNodes; k++) {
                // Subtract 1 because MSH is 1-indexed, and the first entry is the element index [etag ntag1 ntag2 ntag3 ntag4]
                // Map node tags to vertex indices
                int vertexIdx = nodeTagToIdx[block.data[j*(numEleNodes+1) + k + 1]];
                elements(eleIdx+j, k) = vertexIdx;
            }
        }
        // Prepare next block of hexahedra
        eleIdx += block.num_elements_in_block;
    }
    // TODO: Mesh with multiple subcomponents.
    return true;
}


std::tuple<std::vector<std::vector<std::vector<int>>>, std::vector<std::vector<int>>> IO::extract_tet_surfaces (Matrix<double, -1, 3>& vertices, Matrix<int, -1, 4>& eleIdx, bool verboseFlag) {
    // Finds the surfaces of the mesh, and groups them into groups of connected surfaces. We want to guarantee that with the given faceIdx ordering, our normal direction always points outwards.

    std::vector<std::vector<int>> surfaceFaceVertexIdx;    // Surface vertex indices of triangles. Indices sorted in ascending order.
    std::vector<Vector3d> eleCOM;   // Center of mass of the element that the face belongs to. Surface elements only belong to a single element.
    std::set<std::vector<int>> seenIdx;    // Just used to check duplicates
    std::vector<int> faceEle;   // Which element the face belongs to

    // Loop over all element faces, and find the ones that are on the surface
    for (int i = 0; i < eleIdx.rows(); i++) {
        // Get the element vertex indices
        VectorXi ele = eleIdx.row(i);

        // Get the element faces for tetrahedral elements
        Matrix<int, 4, 3> faces;
        faces << 0, 1, 2,
                    0, 1, 3,
                    0, 2, 3,
                    1, 2, 3;
                    
        // Loop over all faces
        for (int j = 0; j < 4; j++) {
            // Get the face vertex indices of triangle
            Vector3i face = faces.row(j);
            std::vector<int> faceIdx = {ele(face(0)), ele(face(1)), ele(face(2))};

            // Sort the face indices
            std::sort(faceIdx.begin(), faceIdx.end());

            // Insert the face indices into the set, if it isn't a duplicate, also insert into array.
            if (seenIdx.insert(faceIdx).second) {
                surfaceFaceVertexIdx.push_back(faceIdx);

                // Calculate center of mass of the element this face belongs to. 
                Vector3d centerOfMass = Vector3d::Zero();
                for (int k = 0; k < ele.size(); k++) {
                    centerOfMass += vertices.row(ele(k));
                }
                centerOfMass /= ele.size();
                eleCOM.push_back(centerOfMass);
                faceEle.push_back(i);
            }
            // If it is a duplicate, we remove it from the set and the array.
            else {
                seenIdx.erase(faceIdx);
                // Find the index of the face in the array and remove it.
                auto it = std::find(surfaceFaceVertexIdx.begin(), surfaceFaceVertexIdx.end(), faceIdx);
                auto it2 = std::distance(surfaceFaceVertexIdx.begin(), it);
                surfaceFaceVertexIdx.erase(it);
                eleCOM.erase(eleCOM.begin() + it2);
                faceEle.erase(faceEle.begin() + it2);
            }
        }
    }

    if (verboseFlag) {
        std::cout << "Number of surface triangles: " << surfaceFaceVertexIdx.size() << std::endl;
    }

    // Split surface faces into groups.
    std::vector<std::vector<std::vector<int>>> surfaceGroups;    // List of groups of surface faces. Each group is a list of face vertex indices.
    std::vector<std::vector<int>> faceEleGroups;    // List of groups of element indices. Each group is a list of element indices.
    while (!seenIdx.empty()) {
        // We start with a face of vertices, keep a set of vertices in the current surface, and keep adding faces to the group that contain at least one vertex in the current surface. Remove this face from the total set of surface faces.

        // Start a new group of surface indices
        std::vector<std::vector<int>> currentGroup;
        // Start a new set of vertices in the current surface
        std::set<int> currentSurfaceVertices;
        std::vector<int> currentSurfaceEle;

        // Find all faces that contain at least one vertex in the current surface
        for (std::size_t i = 0; i < surfaceFaceVertexIdx.size(); i++) {
            std::vector<int> currentFace = surfaceFaceVertexIdx[i];
            for (std::size_t j = 0; j < currentFace.size(); j++) {
                if (currentSurfaceVertices.empty() || (currentSurfaceVertices.find(currentFace[j]) != currentSurfaceVertices.end())) {
                    // Compute the surface normal sign, and swap face idx if necessary (i.e. when normal points towards center of element)
                    Vector3d normal = (vertices.row(currentFace[1]) - vertices.row(currentFace[0])).cross(vertices.row(currentFace[2]) - vertices.row(currentFace[0]));
                    if (normal.dot(eleCOM[i] - vertices.row(currentFace[0]).transpose()) > 0) {
                        currentGroup.push_back({currentFace[0], currentFace[2], currentFace[1]});
                    }
                    else {
                        currentGroup.push_back(currentFace);
                    }
                    // Add to set of vertices
                    for (std::size_t k = 0; k < currentFace.size(); k++) {
                        currentSurfaceVertices.insert(currentFace[k]);
                    }
                    // Add to element indices
                    currentSurfaceEle.push_back(faceEle[i]);
                    // Remove this face from the total set of surface faces.
                    seenIdx.erase(currentFace);
                    surfaceFaceVertexIdx.erase(surfaceFaceVertexIdx.begin() + i);
                    eleCOM.erase(eleCOM.begin() + i);
                    faceEle.erase(faceEle.begin() + i);
                    // As long as we find a single vertex in the current surface, we add the face to the group and are done with this face.
                    // But this means we need to start over with the loop, because the vertices in the current surface have changed. i will be incremented by for loop.
                    i = -1;
                    break;
                }
            }
        }
        // If we reach this point, we have gone through all faces and found no faces that contain a vertex in the current surface. 
        if (verboseFlag) {
            std::cout << "Group size: " << currentGroup.size() << std::endl;
            std::cout << "Remaining faces: " << seenIdx.size() << std::endl;        
        }
        surfaceGroups.push_back(currentGroup);
        faceEleGroups.push_back(currentSurfaceEle);
    }
    return std::make_tuple(surfaceGroups, faceEleGroups);
}


std::tuple<std::vector<std::vector<std::vector<int>>>, std::vector<std::vector<int>>> IO::extract_hex_surfaces(Matrix<double, -1, 3>& vertices, Matrix<int, -1, 8>& eleIdx, bool verboseFlag) {
    std::vector<std::vector<int>> surfaceFaceVertexIdx;
    std::vector<Vector3d> eleCOM;
    std::set<std::vector<int>> seenIdx;
    std::vector<int> faceEle;

    for (int i = 0; i < eleIdx.rows(); i++) {
        VectorXi ele = eleIdx.row(i);
        Matrix<int, 6, 4> faces;
        faces << 0, 1, 2, 3,
                 4, 5, 6, 7,
                 0, 1, 5, 4,
                 1, 2, 6, 5,
                 2, 3, 7, 6,
                 3, 0, 4, 7;
        for (int j = 0; j < 6; j++) {
            std::vector<int> faceIdx = {ele(faces(j,0)), ele(faces(j,1)), ele(faces(j,2)), ele(faces(j,3))};
            // Important: now we create a copy of faceIdx and we sort the copy --> we need this only to check if it's already been inserted
            // We don't want to sort faceIdx as it might not be a quadrilateral anymore. For tets the order doesn't matter, for hex it does.
            auto sorted_faceIdx = faceIdx;
            std::sort(sorted_faceIdx.begin(), sorted_faceIdx.end());
            if (seenIdx.insert(sorted_faceIdx).second) {
                surfaceFaceVertexIdx.push_back(faceIdx);
                Vector3d centerOfMass = Vector3d::Zero();
                for (int k = 0; k < ele.size(); k++) centerOfMass += vertices.row(ele(k)); // not the real center of mass --> but should not be a problem
                centerOfMass /= ele.size();
                eleCOM.push_back(centerOfMass);
                faceEle.push_back(i);
            } else {
                seenIdx.erase(sorted_faceIdx);
                int idx = -1;
                // We cannot use 'find' anymore as with tets because the faces in surfaceFaceVertexIdx are not sorted. We have to sort each element manually and check what is the index of the face by comparing it with sorted_faceIdx
                // It could be improved using an unordered_map
                for (int m = 0; m < (int)surfaceFaceVertexIdx.size(); ++m) {
                    std::vector<int> kk = surfaceFaceVertexIdx[m];
                    std::sort(kk.begin(), kk.end());
                    if (kk == sorted_faceIdx) { 
                        idx = m;
                        break;
                    }
                }
                surfaceFaceVertexIdx.erase(surfaceFaceVertexIdx.begin() + idx);
                eleCOM.erase(eleCOM.begin() + idx);
                faceEle.erase(faceEle.begin() + idx);
            }
        }
    }

    if (verboseFlag) {
        std::cout << "Number of surface quads: " << surfaceFaceVertexIdx.size() << std::endl;
    }

    std::vector<std::vector<std::vector<int>>> surfaceGroups;
    std::vector<std::vector<int>> faceEleGroups;

    while (!surfaceFaceVertexIdx.empty()) {
        std::vector<std::vector<int>> currentGroup;
        std::set<int>                 currentSurfaceVertices;
        std::vector<int>              currentSurfaceEle;

        for (std::size_t i = 0; i < surfaceFaceVertexIdx.size(); ++i) {
            const std::vector<int>& cf = surfaceFaceVertexIdx[i];

            bool touches = currentSurfaceVertices.empty();
            if (!touches) {
                for (int v : cf) if (currentSurfaceVertices.count(v)) { touches = true; break; }
            }
            if (touches) {
                // Making sure that the normal points outward
                Vector3d a = vertices.row(cf[0]).transpose();
                Vector3d b = vertices.row(cf[1]).transpose();
                Vector3d c = vertices.row(cf[2]).transpose();
                Vector3d n = (b - a).cross(c - a);

                std::vector<int> face = cf;
                if (n.dot(eleCOM[i] - a) > 0.0) std::swap(face[1], face[3]);

                currentGroup.push_back(face);
                for (int v : cf) currentSurfaceVertices.insert(v);
                currentSurfaceEle.push_back(faceEle[i]);

                surfaceFaceVertexIdx.erase(surfaceFaceVertexIdx.begin() + i);
                eleCOM.erase(eleCOM.begin() + i);
                faceEle.erase(faceEle.begin() + i);
                i = -1;
            }
        }
        if (verboseFlag) {
            std::cout << "Group size: " << currentGroup.size() << std::endl;
            std::cout << "Remaining faces: " << surfaceFaceVertexIdx.size() << std::endl;
        }
        surfaceGroups.push_back(currentGroup);
        faceEleGroups.push_back(currentSurfaceEle);
    }
    return std::make_tuple(surfaceGroups, faceEleGroups);
}


std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> IO::global_to_local_surfaces (Matrix<int, -1, 4>& elements, std::vector<std::vector<std::vector<int>>>& surfaceGroups, std::vector<std::vector<int>>& faceEleGroups) {
    // Assign surface energies to the surface groups
    // Cast to double vectors since that is what the parameter map expects.
    std::vector<std::vector<double>> surfaceFaces(elements.rows());
    std::vector<std::vector<double>> surfaceIdx(elements.rows());
    
    // Multiple faces per element and each face could belong to a different group.
    for (std::size_t i = 0; i < surfaceGroups.size(); i++) {
        for (std::size_t j = 0; j < surfaceGroups[i].size(); j++) {
            // Flatten the surface face vertex indices. While reconstructing the MatrixXi, we need to know surface face shapes. 
            for (std::size_t k = 0; k < surfaceGroups[i][j].size(); k++) {\
                // Convert to local element face indices (computation is always local).
                for (Index l = 0; l < elements.row(faceEleGroups[i][j]).size(); l++) {
                    if (elements(faceEleGroups[i][j], l) == surfaceGroups[i][j][k]) {
                        surfaceFaces[faceEleGroups[i][j]].push_back((double) l);
                        break;
                    }
                }
            }
            surfaceIdx[faceEleGroups[i][j]].push_back((double) i);
        }
    }
    return std::make_tuple(surfaceFaces, surfaceIdx);
}


std::vector<std::set<int>> IO::get_multimaterial_vertex_tags(const std::string& vertexTagPath) {
    // Initialize the output vector
    std::vector<std::set<int>> vertexTags;
    // Open the file
    std::ifstream infile(vertexTagPath);
    if (!infile) {
        throw std::runtime_error("Error: Could not open file " + vertexTagPath);
    } 
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        int vertex;
        char colon;
        iss >> vertex >> colon;

        if (iss.fail() || colon != ':') {
            throw std::runtime_error("Error: Malformed line in tag file: " + line);
        }
        // Resize the vector if needed
        if (vertex >= static_cast<int>(vertexTags.size())) {
            vertexTags.resize(vertex + 1);
        }
        // Read and insert tags into the set for this vertex
        int tag;
        while (iss >> tag) {
            vertexTags[vertex].insert(tag); // Ensures unique, ordered tags
        }
    }

    infile.close();
    return vertexTags;
}


std::vector<int> IO::get_multimaterial_element_tags(std::vector<std::set<int>> vertexTags, Matrix<int, -1, 4>& elementIdx){
    // Initialize the output vector to store the most frequent tag for each element
    std::vector<int> elementTags(elementIdx.rows(), -1);
    // Loop over all elements
    for (int i = 0; i < elementIdx.rows(); i++) {
        // Map to count tag frequencies
        std::map<int, int> tagFrequency;
        // Loop over all vertices of the element
        for (int j = 0; j < elementIdx.cols(); j++) {
            int vertexIndex = elementIdx(i, j);
            // Get the tags for the current vertex
            const std::set<int>& vertexTagsSet = vertexTags[vertexIndex];
            // Increment the frequency count for each tag
            for (int tag : vertexTagsSet) {
                tagFrequency[tag]++;
            }
        }
        // Find the tag with the maximum frequency
        int maxTag = -1;
        int maxFrequency = 0;
        for (const auto& [tag, frequency] : tagFrequency) {
            if (frequency > maxFrequency) {
                maxTag = tag;
                maxFrequency = frequency;
            }
        }
        // Assign the most frequent tag to the element
        elementTags[i] = maxTag;
    }
    return elementTags;
}


void IO::appendRowToFile(const std::string& filePath, const Eigen::VectorXd& vector) {
    // Convert Eigen::VectorXd to std::vector<double>
    std::vector<double> output(vector.data(), vector.data() + vector.size());
    // Open the file in append mode
    std::ofstream outFile(filePath, std::ios::app);
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open file " << filePath << std::endl;
        return;
    }
    // Write the vector to the file as comma-separated values
    for (size_t i = 0; i < output.size(); ++i) {
        outFile << output[i];
        if (i != output.size() - 1) {
            outFile << ", ";
        }
    }
    // Add a newline at the end of the row
    outFile << std::endl;
    // Close the file
    outFile.close();
    std::cout << "Vector appended successfully to " << filePath << std::endl;
}


Eigen::Matrix<int, -1, 4> IO::hex_to_5_tets (const Eigen::Matrix<int, -1, 8>& hexElements) {
    int numHex = hexElements.rows();
    int numTet = numHex * 5;
    // Empty mesh case
    if (numHex == 0) {
      std::cerr << "Error: empty mesh" << std::endl;
      return Eigen::Matrix<int, -1, 4>(0, 4);
    }
    // Wrong number of columns
    if (hexElements.cols() != 8) {
      std::cerr << "Error: hexahedra must have 8 vertices" << std::endl;
      return Eigen::Matrix<int, -1, 4>(0, 4);
    }
    // Initialize the output matrix
    Eigen::Matrix<int, -1, 4> tetElements(numTet, 4);
    // Convert each hex into 5 tets
    for (int j = 0; j < numHex; ++j) {
        // tet n. 1
        tetElements.row(5*j) << hexElements(j, 0), hexElements(j, 1), hexElements(j, 3), hexElements(j, 4);
        // tet n. 2
        tetElements.row(5*j + 1) << hexElements(j, 1), hexElements(j, 2), hexElements(j, 3), hexElements(j, 6);
        // tet n. 3
        tetElements.row(5*j + 2) << hexElements(j, 1), hexElements(j, 4), hexElements(j, 5), hexElements(j, 6);
        // tet n. 4
        tetElements.row(5*j + 3) << hexElements(j, 3), hexElements(j, 4), hexElements(j, 6), hexElements(j, 7);
        // tet n. 5
        tetElements.row(5*j + 4) << hexElements(j, 1), hexElements(j, 3), hexElements(j, 4), hexElements(j, 6);
    }
    return tetElements;
}


std::pair<Eigen::Matrix<double, -1, 3>, Eigen::Matrix<int, -1, 8>> IO::make_voxel_hex_grid(int nx, int ny, int nz,
                                                                                            double dx, double dy, double dz) {
    if (nx < 1 || ny < 1 || nz < 1) {
        throw std::runtime_error("nx, ny, nz must be >= 1");
    }
    if (dx <= 0.0 || dy <= 0.0 || dz <= 0.0) {
        throw std::runtime_error("dx, dy, dz must be > 0");
    }
    int NX = nx + 1;
    int NY = ny + 1;
    int NZ = nz + 1;
    int numVerts = NX * NY * NZ;
    int numHex = nx * ny * nz;
    Eigen::Matrix<double, -1, 3> vertices(numVerts, 3);
    Eigen::Matrix<int, -1, 8> elements(numHex, 8);

    // vid = i + NX * (j + NY * k)
    for (int k = 0; k < NZ; ++k) {
        double z = k * dz;
        for (int j = 0; j < NY; ++j) {
            double y = j * dy;
            for (int i = 0; i < NX; ++i) {
                double x = i * dx;
                int vid = i + NX * (j + NY * k);
                vertices(vid, 0) = x;
                vertices(vid, 1) = y;
                vertices(vid, 2) = z;
            }
        }
    }
    int e = 0;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int n0 = i + NX * (j + NY * k); // bottom 0
                int n1 = (i + 1) + NX * (j + NY * k); // bottom 1
                int n2 = (i + 1) + NX * ((j + 1) + NY * k); // bottom 2
                int n3 = i + NX * ((j + 1) + NY * k); // bottom 3
                
                int n4 = i + NX * ( j + NY * (k + 1)); // top 4
                int n5 = (i + 1) + NX * (j + NY * (k + 1)); // top 5
                int n6 = (i + 1) + NX * ((j + 1) + NY * (k + 1)); // top 6
                int n7 = i + NX * ((j + 1) + NY * (k + 1)); // top 7
                elements(e,0) = n0;
                elements(e,1) = n1;
                elements(e,2) = n2;
                elements(e,3) = n3;
                elements(e,4) = n4;
                elements(e,5) = n5;
                elements(e,6) = n6;
                elements(e,7) = n7;
                ++e;
            }
        }
    }
    return std::make_pair(vertices, elements);
}

