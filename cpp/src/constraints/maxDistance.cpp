#include "maxDistance.h"

template<int vertexDim, int elementDim>
VectorXd MaxDistance<vertexDim, elementDim>::compute_constraint(const VectorXd& vertices, const VectorXd& verticesPrev, double dt) const {
    
    UNUSED(verticesPrev); UNUSED(dt); 

    // Number of active constraints 
    int m = this->get_num_active_constraints();
    VectorXd constraintVector(m); // Constraint vector

    // Loop over max distance constraint pairs
    int constraintCounter = 0;
    for (unsigned int i = 0; i < this->maxDistanceConstraintPairs_.size(); i++) {
        
        // Extract vertex indices and max distance from the tuple
        int vertex1Index = std::get<0>(this->maxDistanceConstraintPairs_[i]);
        int vertex2Index = std::get<1>(this->maxDistanceConstraintPairs_[i]);
        double maxDistance = std::get<2>(this->maxDistanceConstraintPairs_[i]);

        // Extract vertex positions
        Vector3d pos1 = vertices.segment<3>(3 * vertex1Index);
        Vector3d pos2 = vertices.segment<3>(3 * vertex2Index);

        // Compute the constraint function value
        constraintVector(constraintCounter) = maxDistance - (pos1 - pos2).norm();
        constraintCounter++;
    }
   
    // Assert that the constraint vector has the correct size
    if (constraintCounter != m) {
        std::cerr << bcolors.FAIL << "Error: constraintCounter " << constraintCounter << " does not match the number of active constraints " << m << "in the maxDistance constraint computation." << bcolors.ENDC << std::endl;
        assert(constraintCounter == m);
    }
    return constraintVector;
}


template<int vertexDim, int elementDim>
std::vector<Triplet<double>> MaxDistance<vertexDim, elementDim>::compute_constraint_gradient(const VectorXd& vertices, const VectorXd& verticesPrev, double dt) const {

    UNUSED(verticesPrev); UNUSED(dt); 

    // Number of active constraints (works since only x-coordinate of active vertex is set to 1 in the activeConstraintsMask_)
    int m = this->get_num_active_constraints(); 
    std::vector<Triplet<double>> constraintGradientTriplets; // Triplet list for sparse matrix (m x 3n)

    // Loop over max distance constraint pairs
    int constraintCounter = 0;
    for (unsigned int i = 0; i < this->maxDistanceConstraintPairs_.size(); i++) {
        
        // Extract vertex indices and max distance from the tuple
        int vertex1Index = std::get<0>(this->maxDistanceConstraintPairs_[i]);
        int vertex2Index = std::get<1>(this->maxDistanceConstraintPairs_[i]);

        // Extract vertex positions
        Vector3d pos1 = vertices.segment<3>(3 * vertex1Index);
        Vector3d pos2 = vertices.segment<3>(3 * vertex2Index);

        // Compute the gradient of the constraint function
        Vector3d gradient = (pos1 - pos2).normalized();

        // Add the triplets for the gradient
        constraintGradientTriplets.emplace_back(constraintCounter, 3 * vertex1Index, -gradient(0)); // del g_k / del x_i for vertex1
        constraintGradientTriplets.emplace_back(constraintCounter, 3 * vertex1Index + 1, -gradient(1));
        constraintGradientTriplets.emplace_back(constraintCounter, 3 * vertex1Index + 2, -gradient(2));
        constraintGradientTriplets.emplace_back(constraintCounter, 3 * vertex2Index, gradient(0)); // del g_k / del x_j for vertex2
        constraintGradientTriplets.emplace_back(constraintCounter, 3 * vertex2Index + 1, gradient(1));
        constraintGradientTriplets.emplace_back(constraintCounter, 3 * vertex2Index + 2, gradient(2));
        constraintCounter++;
    }
    // Assert that the number of triplets matches the number of active constraints
    if (constraintCounter != m) {
        std::cerr << bcolors.FAIL << "Error: constraintCounter " << constraintCounter << " does not match the number of active constraints " << m << "in the maxDistance constraint gradient computation." << bcolors.ENDC << std::endl;
        assert(constraintCounter == m);
    }
    return constraintGradientTriplets;
}


template<int vertexDim, int elementDim>
bool MaxDistance<vertexDim, elementDim>::update_active_constraints(const VectorXd& vertices) {
    UNUSED(vertices); 
    // Since max distance constraints are always fixed, we do not need to update the active constraints
    return true;
}

template class MaxDistance <3, 4>;
template class MaxDistance <3, 8>;
