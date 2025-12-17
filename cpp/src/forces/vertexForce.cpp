#include "vertexForce.h"

template<int vertexDim, int elementDim>
VectorXd VertexForce<vertexDim, elementDim>::compute_force (const VectorXd& q, const VectorXd& actuation) const {
    UNUSED(q); 
    // Actuation has size 3n (n the number of vertices), with force triplets for each vertex.
    // This is exactly the force vector that will be added to the gradient vector.
    return actuation;
}