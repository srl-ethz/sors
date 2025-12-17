#include "neumannBC.h"

template<int vertexDim, int elementDim>
VectorXd NeumannBC<vertexDim, elementDim>::compute_constraint(const VectorXd& vertices, const VectorXd& verticesPrev, double dt) const {

    int m = this->get_num_active_constraints(); // Number of active constraints
    VectorXd constraintVector(m); // Constraint vector

    // Loop over all active constraints
    int constraintCounter = 0;
    for (int i = 0; i < 3*this->numVertices_; i++) {
        if (this->neumannBCmask_(i) == 1) { 
            // Compute the constraint function value
            constraintVector(constraintCounter) = vertices(i) - verticesPrev(i) - dt * this->neumannBCvalue_(i);
            constraintCounter++;
        }
    }
    return constraintVector; 
}


template<int vertexDim, int elementDim>
std::vector<Triplet<double>> NeumannBC<vertexDim, elementDim>::compute_constraint_gradient(const VectorXd& vertices, const VectorXd& verticesPrev, double dt) const {

    UNUSED(vertices); UNUSED(verticesPrev); UNUSED(dt);

    std::vector<Triplet<double>> constraintGradientTriplets; // Triplet list for sparse matrix (m x 3n)
    // Loop over all active constraints
    int constraintCounter = 0;
    // If constraint is active, add a row with 1 in the corresponding column
    for (int i = 0; i < 3*this->numVertices_; i++) {
        if (this->neumannBCmask_(i) == 1) { 
            // Add a row with 1 in the corresponding column
            constraintGradientTriplets.emplace_back(constraintCounter, i, 1.0); // del g_k / del x_i = 1
            constraintCounter++;
        }
    }
 
    return constraintGradientTriplets;
}


template<int vertexDim, int elementDim>
bool NeumannBC<vertexDim, elementDim>::update_active_constraints(const VectorXd& vertices) {
    UNUSED(vertices);
    // Since Neumann boundary conditions are always fixed, we do not need to update the active constraints
    return true;
}

template class NeumannBC <3, 4>;
template class NeumannBC <3, 8>;
