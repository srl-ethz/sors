#ifndef HYDROFORCE_H
#define HYDROFORCE_H

#include "externalForce.h"

template<int vertexDim, int elementDim>
class HydroForce : public ExternalForce<vertexDim, elementDim> {
public:
    HydroForce(MatrixXi surfaceVertexIdx, double fluidDensity, double cD, double cT)
        : surfaceVertexIdx_(surfaceVertexIdx),
          fluidDensity_(fluidDensity),
          cD_(cD),
          cT_(cT) {
        this->actuationFlag_ = true;
    }

    VectorXd compute_force(
        const VectorXd& q,
        const VectorXd& v,
        const VectorXd& actuation) const override;

    std::vector<Triplet<double>> compute_force_gradient(
        const VectorXd& q,
        const VectorXd& v,
        const VectorXd& actuation,
        double dvdx) const override;

    VectorXd compute_force(const VectorXd& q, const VectorXd& actuation) const override {
        return compute_force(q, VectorXd::Zero(q.size()), actuation);
    }

    std::string get_force_name() const override {
        return "hydro";
    }

private:
    const MatrixXi surfaceVertexIdx_;
    const double fluidDensity_;
    const double cD_;
    const double cT_;
};

#endif
