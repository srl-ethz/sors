#ifndef TETRAHEDRON_H
#define TETRAHEDRON_H

#include "element.h"

static const int TET_E_DIM = 4;
static const int TET_V_DIM = 3;
static const int TET_EV_DIM = TET_V_DIM * TET_E_DIM;

/**
 * @class Tetrahedron
 * @brief Linear 4-node tetrahedral finite element.
 *
 * Specializes Element<3,4> and provides tetrahedron-specific volume, deformation
 * gradient, and center-of-mass computations, as well as setup of all attached
 * ElementEnergy models for this element.
 */
class Tetrahedron : public Element<TET_V_DIM, TET_E_DIM> {
public:
    /**
     * @brief Construct a tetrahedral element from undeformed geometry and parameters.
     *
     * Builds the inverse reference shape matrix, precomputes the deformation
     * Hessian, initializes unit matrices, and creates
     * all requested ElementEnergy instances based on the provided parameter map
     * and energy-type set.
     */
    Tetrahedron( 
        Params parameterMap,
        const Matrix<double, TET_E_DIM, TET_V_DIM>& undeformedVertices,
        const Vector<double, TET_V_DIM>& gravAcceleration,
        std::set<std::string> elementEnergiesStringSet
    );

    Tetrahedron( 
        Params parameterMap,
        const Matrix<double, TET_E_DIM, TET_V_DIM>& undeformedVertices,
        std::set<std::string> elementEnergiesStringSet
    ) : Tetrahedron(parameterMap, undeformedVertices, Vector<double, TET_V_DIM>::Zero(), elementEnergiesStringSet) {}

    // Computes gradient of the energy function with respect to vertices
    Vector<double, TET_EV_DIM> compute_gradient(
        Matrix<double, TET_E_DIM, TET_V_DIM>& vertices,
        Params& actuation) const override;

    // Computes Hessian matrix of the energy function with respect to vertices
    Matrix<double, TET_EV_DIM, TET_EV_DIM> compute_hessian(
        Matrix<double, TET_E_DIM, TET_V_DIM>& vertices,
        Params& actuation) const override;

private:
    // 3x3, inverted reference shape matrix for tetrahedron
    Matrix<double, TET_V_DIM, TET_V_DIM> invRefShapeMatrix_; 
    
    // Computes volume of the tetrahedron
    double compute_volume(const Matrix<double, TET_E_DIM, TET_V_DIM>& vertices) const override;

    // Computes deformation gradient of the tetrahedron
    std::vector<Matrix<double, TET_V_DIM, TET_V_DIM>> compute_deformation_gradients(Matrix<double, TET_E_DIM, TET_V_DIM>& vertices) const override;

    // Computes center of mass of tetrahedron
    Vector<double, TET_V_DIM> compute_com(const Matrix<double, TET_E_DIM, TET_V_DIM>& vertices) const override;
};

#endif
