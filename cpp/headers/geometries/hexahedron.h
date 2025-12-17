#ifndef HEXAHEDRON_H
#define HEXAHEDRON_H

#include "element.h"

static const int HEX_E_DIM = 8;
static const int HEX_V_DIM = 3;
static const int HEX_EV_DIM = HEX_V_DIM * HEX_E_DIM;

/**
 * @class Hexahedron
 * @brief Linear 8-node hexahedral finite element using trilinear shape functions.
 *
 * Specializes Element<3,8> and provides hex-specific volume, deformation
 * gradient / Hessian, and center-of-mass computations, as well as setup of all
 * attached ElementEnergy models based on the chosen constitutive laws.
 */
class Hexahedron : public Element<HEX_V_DIM, HEX_E_DIM> {
public:
    /**
     * @brief Construct a hexahedral element from undeformed geometry and parameters.
     *
     * Initializes shape function gradients and corresponding deformation Hessians
     * at Gauss quadrature points, and instantiates all requested ElementEnergy 
     * objects from the parameter map and energy-type set.
     */
    Hexahedron(
        Params parameterMap,
        const Matrix<double, HEX_E_DIM, HEX_V_DIM>& undeformedVertices,
        std::set<std::string> elementEnergiesStringSet
    );

    // Computes gradient of the energy function with respect to vertices
    Vector<double, HEX_EV_DIM> compute_gradient(
        Matrix<double, HEX_E_DIM, HEX_V_DIM>& vertices,
        Params& actuation) const override;
    
    // Computes Hessian matrix of the energy function with respect to vertices
    Matrix<double, HEX_EV_DIM, HEX_EV_DIM> compute_hessian(
        Matrix<double, HEX_E_DIM, HEX_V_DIM>& vertices,
        Params& actuation) const override;

private:
    // 8 8x3 Matrixes that contains the gradient of the shape functions. Used to compute F and nablaF
    // NB: it is important that shape_function_gradients_ is declared before deformationHessians_, 
    // otherwise the initializer list of the constructor doesn't work since deformationHessians_ is initialized using shapeFunctionsGradients_
    const std::vector<Matrix<double, HEX_E_DIM, HEX_V_DIM>> shapeFunctionsGradients_;
    
    // Computes volume of the hexahedron
    double compute_volume(const Matrix<double, HEX_E_DIM, HEX_V_DIM>& vertices) const override;

    // Computes deformation gradients of the hexahedron over the quadrature points
    std::vector<Matrix<double, HEX_V_DIM, HEX_V_DIM>> compute_deformation_gradients(Matrix<double, HEX_E_DIM, HEX_V_DIM>& vertices) const override;

    // Computes deformation hessian of the hexahedron
    std::vector<Matrix<double, HEX_V_DIM*HEX_V_DIM, HEX_V_DIM*HEX_E_DIM>> compute_deformation_hessians();

    // Computes the gradients of the shape functions: shape_functions_gradients
    std::vector<Matrix<double, HEX_E_DIM, HEX_V_DIM>> compute_shape_functions_gradients(const Matrix<double, HEX_E_DIM, HEX_V_DIM>& undeformedVert);

    // Computes center of mass of hexahedron
    Vector<double, HEX_V_DIM> compute_com(const Matrix<double, HEX_E_DIM, HEX_V_DIM>& vertices) const override;
};

#endif
