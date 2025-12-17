#include "pressureForce.h"

template<int vertexDim, int elementDim>
PressureForce<vertexDim, elementDim>::PressureForce(
    MatrixXi surfaceVertexIdx, // List of surface elements (for example (-1, 3) for triangle surface)
    VectorXi surfaceGroups     // List of which actuation group each surface element belongs to
) : surfaceVertexIdx_(surfaceVertexIdx), surfaceGroups_(surfaceGroups) {
    this->actuationFlag_ = true; // Set the actuation flag to true, as pressure force requires actuation
    assert(surfaceVertexIdx.rows() == surfaceGroups.rows()); // Ensure that each triangle has a corresponding actuation group
}


template<int vertexDim, int elementDim>
VectorXd PressureForce<vertexDim, elementDim>::compute_force (const VectorXd& q, const VectorXd& actuation) const {
    // Actuation has size numGroups, where each surface triangle belongs to a certain actuation group defined in surfaceGroups.
    MatrixXd vertices = q.reshaped<RowMajor>(q.size() / vertexDim, vertexDim); // Reshape the vector q into a matrix of vertices
    VectorXd pressureForces = VectorXd::Zero(vertices.size());
    
    // Loop over all faces that are part of the outer surface that will experience a force.
    #pragma omp parallel for reduction(+:pressureForces)
    for (int i = 0; i < this->surfaceVertexIdx_.rows(); i++) {
        const VectorXi face = this->surfaceVertexIdx_.row(i);

        // Compute the area of the face
        Vector3d area = Vector3d::Zero();
        if (elementDim == 4 && vertexDim == 3) {
            // Triangle surface
            const Vector3d v1 = vertices.row(face[1]) - vertices.row(face[0]);
            const Vector3d v2 = vertices.row(face[2]) - vertices.row(face[0]);
            area = 0.5 * v1.cross(v2);
        } else if (elementDim == 8 && vertexDim == 3) {
            const Vector3d v1 = vertices.row(face[1]) - vertices.row(face[0]);
            const Vector3d v2 = vertices.row(face[3]) - vertices.row(face[0]);
            area = 0.5 * v1.cross(v2);
            const Vector3d v3 = vertices.row(face[3]) - vertices.row(face[2]);
            const Vector3d v4 = vertices.row(face[1]) - vertices.row(face[2]);
            area += 0.5 * v3.cross(v4); // Sum of two triangles area
        } else {
            // Not yet implemented
            assert(false);
        }

        // Compute the force on each vertex of the face
        assert(actuation.row(this->surfaceGroups_(i)).size() == 1);   // In case of pressure actuation, actuation vector should have size 1
        // Divide pressure force on surface equally to each of the N vertices of the face
        const Vector3d fP = -actuation.row(this->surfaceGroups_(i))(0) * area / face.size();

        // Add the force to each vertex of the face, divide pressure force on surface equally to each of the 3 vertices of the face
        for (int j = 0; j < face.size(); j++) {
            for (int k = 0; k < vertexDim; k++) {
                pressureForces(vertexDim*face[j]+k) += fP(k);
            }
        }
    }
    return pressureForces;
}


template<int vertexDim, int elementDim>
std::vector<Triplet<double>> PressureForce<vertexDim, elementDim>::compute_force_gradient (const VectorXd& q, const VectorXd& actuation) const {

    MatrixXd vertices = q.reshaped<RowMajor>(q.size() / vertexDim, vertexDim); // Reshape the vector q into a matrix of vertices
    std::vector<Triplet<double>> forceDerivativeTriplets(surfaceVertexIdx_.rows() * surfaceVertexIdx_.cols() * surfaceVertexIdx_.cols() * vertexDim * vertexDim); // Preallocate space for triplets
    // Loop over all faces that are part of the outer surface that will experience a force.
    #pragma omp parallel for
    for (int i = 0; i < this->surfaceVertexIdx_.rows(); i++) {
        const VectorXi face = this->surfaceVertexIdx_.row(i);

        // Compute the triplet entries
        if (elementDim == 4 && vertexDim == 3) {
            // Triangle surface
            const Vector3d x0 = vertices.row(face[0]);
            const Vector3d x1 = vertices.row(face[1]);
            const Vector3d x2 = vertices.row(face[2]);

            const Matrix<double, vertexDim, vertexDim> hessX0 = (Matrix<double, vertexDim, vertexDim>() << 
                0, -x2[2]+x1[2], x2[1]-x1[1],
                x2[2]-x1[2], 0, -x2[0]+x1[0],
                -x2[1]+x1[1], x2[0]-x1[0], 0
            ).finished();
            const Matrix<double, vertexDim, vertexDim> hessX1 = (Matrix<double, vertexDim, vertexDim>() << 
                0, -x0[2]+x2[2], x0[1]-x2[1],
                x0[2]-x2[2], 0, -x0[0]+x2[0],
                -x0[1]+x2[1], x0[0]-x2[0], 0
            ).finished();
            const Matrix<double, vertexDim, vertexDim> hessX2 = (Matrix<double, vertexDim, vertexDim>() << 
                0, -x1[2]+x0[2], x1[1]-x0[1],
                x1[2]-x0[2], 0, -x1[0]+x0[0],
                -x1[1]+x0[1], x1[0]-x0[0], 0
            ).finished();

            std::vector<MatrixXd> hessX = {hessX0, hessX1, hessX2};

            // Create triplets for force derivative
            const double pressure = actuation.row(this->surfaceGroups_(i))(0);

            for (int j = 0; j < face.size(); j++) {
                for (int k = 0; k < face.size(); k++) {
                    for (int l = 0; l < vertexDim; l++) {
                        for (int m = 0; m < vertexDim; m++) {
                            forceDerivativeTriplets[i*face.size()*face.size()*vertexDim*vertexDim + j*face.size()*vertexDim*vertexDim + k*vertexDim*vertexDim + l*vertexDim + m] \
                            = Triplet<double>(
                                vertexDim * face[j] + l, 
                                vertexDim * face[k] + m, 
                                pressure / 6 * hessX[k](l, m)
                            );
                        }
                    }
                }
            }
        } else if (elementDim == 8 && vertexDim == 3) {
            // Quad surface split in two triangles
            Vector3d quadx0 = vertices.row(face[0]);
            Vector3d quadx1 = vertices.row(face[1]);
            Vector3d quadx2 = vertices.row(face[2]);
            Vector3d quadx3 = vertices.row(face[3]);
            Matrix<double, vertexDim, vertexDim> hessX0, hessX1, hessX2, hessX3;
            Vector3d x0, x1, x2; // Auxiliary vectors to make computations using the same expressions for the tets

            // Triangle 1
            x0 = quadx0;
            x1 = quadx1;
            x2 = quadx3;
            hessX0 = (Matrix<double, vertexDim, vertexDim>() << 
                0, -x2[2]+x1[2], x2[1]-x1[1],
                x2[2]-x1[2], 0, -x2[0]+x1[0],
                -x2[1]+x1[1], x2[0]-x1[0], 0
            ).finished();
            hessX1 = (Matrix<double, vertexDim, vertexDim>() << 
                0, -x0[2]+x2[2], x0[1]-x2[1],
                x0[2]-x2[2], 0, -x0[0]+x2[0],
                -x0[1]+x2[1], x0[0]-x2[0], 0
            ).finished();
            hessX2 << 0, 0, 0, 0, 0, 0, 0, 0, 0;
            hessX3 = (Matrix<double, vertexDim, vertexDim>() << 
                0, -x1[2]+x0[2], x1[1]-x0[1],
                x1[2]-x0[2], 0, -x1[0]+x0[0],
                -x1[1]+x0[1], x1[0]-x0[0], 0
            ).finished();
            std::vector<MatrixXd> hessX = {hessX0, hessX1, hessX2, hessX3};

            // Triangle 2
            x0 = quadx2;
            x1 = quadx3;
            x2 = quadx1;
            hessX2 = (Matrix<double, vertexDim, vertexDim>() << 
                0, -x2[2]+x1[2], x2[1]-x1[1],
                x2[2]-x1[2], 0, -x2[0]+x1[0],
                -x2[1]+x1[1], x2[0]-x1[0], 0
            ).finished();
            hessX3 = (Matrix<double, vertexDim, vertexDim>() << 
                0, -x0[2]+x2[2], x0[1]-x2[1],
                x0[2]-x2[2], 0, -x0[0]+x2[0],
                -x0[1]+x2[1], x0[0]-x2[0], 0
            ).finished();
            hessX1 = (Matrix<double, vertexDim, vertexDim>() << 
                0, -x1[2]+x0[2], x1[1]-x0[1],
                x1[2]-x0[2], 0, -x1[0]+x0[0],
                -x1[1]+x0[1], x1[0]-x0[0], 0
            ).finished();
            hessX[1] += hessX1, hessX[2] = hessX2, hessX[3] += hessX3;

            // Create triplets for force derivative
            const double pressure = actuation.row(this->surfaceGroups_(i))(0);

            for (int j = 0; j < face.size(); j++) {
                for (int k = 0; k < face.size(); k++) {
                    for (int l = 0; l < vertexDim; l++) {
                        for (int m = 0; m < vertexDim; m++) {
                            forceDerivativeTriplets[i*face.size()*face.size()*vertexDim*vertexDim + j*face.size()*vertexDim*vertexDim + k*vertexDim*vertexDim + l*vertexDim + m] \
                            = Triplet<double>(
                                vertexDim * face[j] + l, 
                                vertexDim * face[k] + m, 
                                pressure / 6 * hessX[k](l, m)
                            );
                        }
                    }
                }
            }
        } else {
            // Not yet implemented
            assert(false);
        }
    }
    return forceDerivativeTriplets;
}