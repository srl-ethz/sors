#include "hydroForce.h"

template<int vertexDim, int elementDim>
VectorXd HydroForce<vertexDim, elementDim>::compute_force(
    const VectorXd& q,
    const VectorXd& v,
    const VectorXd& actuation) const {
    const MatrixXd vertices = q.reshaped<RowMajor>(q.size() / vertexDim, vertexDim);
    const MatrixXd velocity = v.reshaped<RowMajor>(v.size() / vertexDim, vertexDim);
    assert(actuation.col(0).size() == 3);
    const VectorXd vFluid = actuation.col(0);
    VectorXd hydroForces = VectorXd::Zero(vertices.size());

    #pragma omp parallel for reduction(+:hydroForces)
    for (int i = 0; i < surfaceVertexIdx_.rows(); ++i) {
        const VectorXi face = surfaceVertexIdx_.row(i);

        Vector<double, vertexDim> normal = Vector<double, vertexDim>::Zero();
        Vector<double, vertexDim> vSolid = Vector<double, vertexDim>::Zero();
        if (elementDim == 4 && vertexDim == 3) {
            const Vector3d v1 = vertices.row(face[1]) - vertices.row(face[0]);
            const Vector3d v2 = vertices.row(face[2]) - vertices.row(face[0]);
            normal = 0.5 * v1.cross(v2);
            vSolid = (velocity.row(face[0]) + velocity.row(face[1]) + velocity.row(face[2])) / 3;
        }
        else if (elementDim == 8 && vertexDim == 3) {
            const Vector3d v1 = vertices.row(face[1]) - vertices.row(face[0]);
            const Vector3d v2 = vertices.row(face[3]) - vertices.row(face[0]);
            normal = 0.5 * v1.cross(v2);
            const Vector3d v3 = vertices.row(face[3]) - vertices.row(face[2]);
            const Vector3d v4 = vertices.row(face[1]) - vertices.row(face[2]);
            normal += 0.5 * v3.cross(v4);
            vSolid = (velocity.row(face[0]) + velocity.row(face[1]) + velocity.row(face[2]) + velocity.row(face[3])) / 4;
        }
        else {
            assert(false);
        }

        const Vector<double, vertexDim> vRel = vFluid - vSolid;
        const Vector3d fDrag = 0.5 * fluidDensity_ * cD_ * normal.norm() * vRel.norm() * vRel / face.size();
        const Vector3d fThrust = 0.5 * fluidDensity_ * cT_ * vRel.norm() * vRel.norm() * normal / face.size();

        for (int j = 0; j < face.size(); ++j) {
            for (int k = 0; k < vertexDim; ++k) {
                hydroForces(vertexDim * face[j] + k) += fDrag(k);
                hydroForces(vertexDim * face[j] + k) += fThrust(k);
            }
        }
    }
    return hydroForces;
}

template<int vertexDim, int elementDim>
std::vector<Triplet<double>> HydroForce<vertexDim, elementDim>::compute_force_gradient(
    const VectorXd& q,
    const VectorXd& v,
    const VectorXd& actuation,
    double dvdx) const {
    const MatrixXd vertices = q.reshaped<RowMajor>(q.size() / vertexDim, vertexDim);
    const MatrixXd velocity = v.reshaped<RowMajor>(v.size() / vertexDim, vertexDim);
    assert(actuation.col(0).size() == 3);
    const VectorXd vFluid = actuation.col(0);

    std::vector<Triplet<double>> forceDerivativeTriplets(
        surfaceVertexIdx_.rows() * surfaceVertexIdx_.cols() * surfaceVertexIdx_.cols() * vertexDim * vertexDim);

    #pragma omp parallel for
    for (int i = 0; i < surfaceVertexIdx_.rows(); ++i) {
        const VectorXi face = surfaceVertexIdx_.row(i);

        if (elementDim == 4 && vertexDim == 3) {
            const Vector3d x0 = vertices.row(face[0]);
            const Vector3d x1 = vertices.row(face[1]);
            const Vector3d x2 = vertices.row(face[2]);

            const Vector<double, vertexDim> normal = 0.5 * (x1 - x0).cross(x2 - x0);
            const Vector<double, vertexDim> vSolid = (velocity.row(face[0]) + velocity.row(face[1]) + velocity.row(face[2])) / 3;
            assert(normal.norm() > 0);

            const Vector<double, vertexDim> vRel = vFluid - vSolid;
            if (vRel.norm() == 0) {
                continue;
            }

            const Matrix3d x2_x1W = (Matrix3d() <<
                 0,             -x2[2]+x1[2],     x2[1]-x1[1],
                 x2[2]-x1[2],    0,              -x2[0]+x1[0],
                -x2[1]+x1[1],    x2[0]-x1[0],     0
            ).finished();
            const Matrix3d x0_x2W = (Matrix3d() <<
                 0,             -x0[2]+x2[2],     x0[1]-x2[1],
                 x0[2]-x2[2],     0,             -x0[0]+x2[0],
                -x0[1]+x2[1],    x0[0]-x2[0],     0
            ).finished();
            const Matrix3d x1_x0W = (Matrix3d() <<
                 0,             -x1[2]+x0[2],     x1[1]-x0[1],
                 x1[2]-x0[2],    0,              -x1[0]+x0[0],
                -x1[1]+x0[1],    x1[0]-x0[0],     0
            ).finished();

            const Matrix3d hessDragX0 = 0.5 * fluidDensity_ * cD_ / face.size() * (
                0.5 * vRel.norm() / normal.norm() * vRel * (normal.transpose() * x2_x1W)
                - normal.norm() / (3 * vRel.norm()) * dvdx * vRel * vRel.transpose()
                - normal.norm() * vRel.norm() / 3 * dvdx * Matrix3d::Identity()
            );
            const Matrix3d hessDragX1 = 0.5 * fluidDensity_ * cD_ / face.size() * (
                0.5 * vRel.norm() / normal.norm() * vRel * (normal.transpose() * x0_x2W)
                - normal.norm() / (3 * vRel.norm()) * dvdx * vRel * vRel.transpose()
                - normal.norm() * vRel.norm() / 3 * dvdx * Matrix3d::Identity()
            );
            const Matrix3d hessDragX2 = 0.5 * fluidDensity_ * cD_ / face.size() * (
                0.5 * vRel.norm() / normal.norm() * vRel * (normal.transpose() * x1_x0W)
                - normal.norm() / (3 * vRel.norm()) * dvdx * vRel * vRel.transpose()
                - normal.norm() * vRel.norm() / 3 * dvdx * Matrix3d::Identity()
            );

            const Matrix3d hessThrustX0 = 0.5 * fluidDensity_ * cT_ / face.size() * (
                0.5 * vRel.norm() * vRel.norm() * x2_x1W
                - 2 / 3 * dvdx * normal * vRel.transpose()
            );
            const Matrix3d hessThrustX1 = 0.5 * fluidDensity_ * cT_ / face.size() * (
                0.5 * vRel.norm() * vRel.norm() * x0_x2W
                - 2 / 3 * dvdx * normal * vRel.transpose()
            );
            const Matrix3d hessThrustX2 = 0.5 * fluidDensity_ * cT_ / face.size() * (
                0.5 * vRel.norm() * vRel.norm() * x1_x0W
                - 2 / 3 * dvdx * normal * vRel.transpose()
            );

            const std::vector<MatrixXd> hessX = {
                hessDragX0 + hessThrustX0,
                hessDragX1 + hessThrustX1,
                hessDragX2 + hessThrustX2
            };

            for (int j = 0; j < face.size(); ++j) {
                for (int k = 0; k < face.size(); ++k) {
                    for (int l = 0; l < vertexDim; ++l) {
                        for (int m = 0; m < vertexDim; ++m) {
                            forceDerivativeTriplets[
                                i * face.size() * face.size() * vertexDim * vertexDim
                                + j * face.size() * vertexDim * vertexDim
                                + k * vertexDim * vertexDim
                                + l * vertexDim + m] =
                                Triplet<double>(vertexDim * face[j] + l, vertexDim * face[k] + m, hessX[k](l, m));
                        }
                    }
                }
            }
        }
        else if (elementDim == 8 && vertexDim == 3) {
            const Vector3d x0 = vertices.row(face[0]);
            const Vector3d x1 = vertices.row(face[1]);
            const Vector3d x2 = vertices.row(face[2]);
            const Vector3d x3 = vertices.row(face[3]);

            const Vector<double, vertexDim> normal = 0.5 * ((x1 - x0).cross(x3 - x0) + (x3 - x2).cross(x1 - x2));
            const Vector<double, vertexDim> vSolid = (velocity.row(face[0]) + velocity.row(face[1]) + velocity.row(face[2]) + velocity.row(face[3])) / 4;
            assert(normal.norm() > 0);

            const Vector<double, vertexDim> vRel = vFluid - vSolid;
            if (vRel.norm() == 0) {
                continue;
            }

            const Matrix3d x3_x1W = (Matrix3d() <<
                 0,             -x3[2]+x1[2],     x3[1]-x1[1],
                 x3[2]-x1[2],    0,              -x3[0]+x1[0],
                -x3[1]+x1[1],    x3[0]-x1[0],     0
            ).finished();
            const Matrix3d x0_x3W = (Matrix3d() <<
                 0,             -x0[2]+x3[2],     x0[1]-x3[1],
                 x0[2]-x3[2],     0,             -x0[0]+x3[0],
                -x0[1]+x3[1],    x0[0]-x3[0],     0
            ).finished();
            const Matrix3d x3_x2W = (Matrix3d() <<
                 0,             -x3[2]+x2[2],     x3[1]-x2[1],
                 x3[2]-x2[2],     0,             -x3[0]+x2[0],
                -x3[1]+x2[1],    x3[0]-x2[0],     0
            ).finished();
            const Matrix3d x1_x3W = (Matrix3d() <<
                 0,             -x1[2]+x3[2],     x1[1]-x3[1],
                 x1[2]-x3[2],     0,             -x1[0]+x3[0],
                -x1[1]+x3[1],    x1[0]-x3[0],     0
            ).finished();
            const Matrix3d x1_x0W = (Matrix3d() <<
                 0,             -x1[2]+x0[2],     x1[1]-x0[1],
                 x1[2]-x0[2],    0,              -x1[0]+x0[0],
                -x1[1]+x0[1],    x1[0]-x0[0],     0
            ).finished();
            const Matrix3d x2_x1W = (Matrix3d() <<
                 0,             -x2[2]+x1[2],     x2[1]-x1[1],
                 x2[2]-x1[2],    0,              -x2[0]+x1[0],
                -x2[1]+x1[1],    x2[0]-x1[0],     0
            ).finished();

            const Matrix3d hessDragX0 = 0.5 * fluidDensity_ * cD_ / face.size() * (
                0.5 * vRel.norm() / normal.norm() * vRel * (normal.transpose() * x3_x1W)
                - normal.norm() / (3 * vRel.norm()) * dvdx * vRel * vRel.transpose()
                - normal.norm() * vRel.norm() / 3 * dvdx * Matrix3d::Identity()
            );
            const Matrix3d hessDragX1 = 0.5 * fluidDensity_ * cD_ / face.size() * (
                0.5 * vRel.norm() / normal.norm() * vRel * (normal.transpose() * (x0_x3W + x3_x2W))
                - normal.norm() / (3 * vRel.norm()) * dvdx * vRel * vRel.transpose()
                - normal.norm() * vRel.norm() / 3 * dvdx * Matrix3d::Identity()
            );
            const Matrix3d hessDragX2 = 0.5 * fluidDensity_ * cD_ / face.size() * (
                0.5 * vRel.norm() / normal.norm() * vRel * (normal.transpose() * x1_x3W)
                - normal.norm() / (3 * vRel.norm()) * dvdx * vRel * vRel.transpose()
                - normal.norm() * vRel.norm() / 3 * dvdx * Matrix3d::Identity()
            );
            const Matrix3d hessDragX3 = 0.5 * fluidDensity_ * cD_ / face.size() * (
                0.5 * vRel.norm() / normal.norm() * vRel * (normal.transpose() * (x1_x0W + x2_x1W))
                - normal.norm() / (3 * vRel.norm()) * dvdx * vRel * vRel.transpose()
                - normal.norm() * vRel.norm() / 3 * dvdx * Matrix3d::Identity()
            );

            const Matrix3d hessThrustX0 = 0.5 * fluidDensity_ * cT_ / face.size() * (
                0.5 * vRel.norm() * vRel.norm() * x3_x1W
                - 2 / 3 * dvdx * normal * vRel.transpose()
            );
            const Matrix3d hessThrustX1 = 0.5 * fluidDensity_ * cT_ / face.size() * (
                0.5 * vRel.norm() * vRel.norm() * (x0_x3W + x3_x2W)
                - 2 / 3 * dvdx * normal * vRel.transpose()
            );
            const Matrix3d hessThrustX2 = 0.5 * fluidDensity_ * cT_ / face.size() * (
                0.5 * vRel.norm() * vRel.norm() * x1_x3W
                - 2 / 3 * dvdx * normal * vRel.transpose()
            );
            const Matrix3d hessThrustX3 = 0.5 * fluidDensity_ * cT_ / face.size() * (
                0.5 * vRel.norm() * vRel.norm() * (x1_x0W + x2_x1W)
                - 2 / 3 * dvdx * normal * vRel.transpose()
            );

            const std::vector<MatrixXd> hessX = {
                hessDragX0 + hessThrustX0,
                hessDragX1 + hessThrustX1,
                hessDragX2 + hessThrustX2,
                hessDragX3 + hessThrustX3
            };

            for (int j = 0; j < face.size(); ++j) {
                for (int k = 0; k < face.size(); ++k) {
                    for (int l = 0; l < vertexDim; ++l) {
                        for (int m = 0; m < vertexDim; ++m) {
                            forceDerivativeTriplets[
                                i * face.size() * face.size() * vertexDim * vertexDim
                                + j * face.size() * vertexDim * vertexDim
                                + k * vertexDim * vertexDim
                                + l * vertexDim + m] =
                                Triplet<double>(vertexDim * face[j] + l, vertexDim * face[k] + m, hessX[k](l, m));
                        }
                    }
                }
            }
        }
        else {
            assert(false);
        }
    }

    return forceDerivativeTriplets;
}

template class HydroForce<3, 4>;
template class HydroForce<3, 8>;
