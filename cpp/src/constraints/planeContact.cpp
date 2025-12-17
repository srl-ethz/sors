#include "planeContact.h"

template<int vertexDim, int elementDim>
VectorXd PlaneContact<vertexDim, elementDim>::compute_constraint(const VectorXd& vertices, const VectorXd& verticesPrev, double dt) const {
    
    UNUSED(verticesPrev); UNUSED(dt); 

    // Number of active constraints 
    int m = this->get_num_active_constraints();
    VectorXd constraintVector(m);

    // Only loop over surface vertices
    int constraintCounter = 0;
    for (unsigned int i = 0; i < this->surfaceVerticesIdx_.size(); i++) {

        // If no collision, skip this vertex
        if (this->planeCollisionMask_[i] == -1) { 
            continue;
        } else {
            int surfaceVertexIndex = this->surfaceVerticesIdx_(i);
            int planeIndex = this->planeCollisionMask_[i];
            Vector3d vertex = vertices.segment<3>(3*surfaceVertexIndex);
            // Assert if the plane index is valid (it should be less than the number of planes and greater than or equal to 0)
            if (!(planeIndex < int(planes_.size()) && planeIndex >= 0)) {
                std::cerr << bcolors.FAIL << "Error: planeIndex " << planeIndex << " is out of bounds. It must be between 0 and " << planes_.size() - 1 << "." << bcolors.ENDC << std::endl;
                assert(planeIndex < int(planes_.size()) && planeIndex >= 0);
            }
            const Plane3D& plane = planes_[planeIndex];
            Vector3d normal = plane.get_normal();
            Vector3d r0 = plane.get_point();
            // Compute the constraint function 
            constraintVector(constraintCounter) = normal.dot(vertex - r0);
            constraintCounter++;
        }
    }
    // Assert that the constraint vector has the correct size
    if (constraintCounter != m) {
        std::cerr << bcolors.FAIL << "Error: constraintCounter " << constraintCounter << " does not match the number of active constraints " << m << "in the plane contact constraint computation." << bcolors.ENDC << std::endl;
        assert(constraintCounter == m);
    }
    return constraintVector;
}


template<int vertexDim, int elementDim>
std::vector<Triplet<double>> PlaneContact<vertexDim, elementDim>::compute_constraint_gradient(const VectorXd& vertices, const VectorXd& verticesPrev, double dt) const {

    UNUSED(vertices); UNUSED(verticesPrev); UNUSED(dt); 

    int m = this->get_num_active_constraints(); 
    std::vector<Triplet<double>> constraintGradientTriplets; // Triplet list for sparse matrix (m x 3n)
   
    // Loop over all surface vertices
    int constraintCounter = 0;
    for (unsigned int i = 0; i < this->surfaceVerticesIdx_.size(); i++) {
        // If no collision, skip this vertex
        if (this->planeCollisionMask_[i] == -1) { 
            continue;
        } else {
            int surfaceVertexIndex = this->surfaceVerticesIdx_(i);
            int planeIndex = this->planeCollisionMask_[i];
            // Assert if the plane index is valid (it should be less than the number of planes and greater than or equal to 0)
            if (!(planeIndex < int(planes_.size()) && planeIndex >= 0)) {
                std::cerr << bcolors.FAIL << "Error: planeIndex " << planeIndex << " is out of bounds. It must be between 0 and " << planes_.size() - 1 << "." << bcolors.ENDC << std::endl;
                assert(planeIndex < int(planes_.size()) && planeIndex >= 0);
            }
            const Plane3D& plane = planes_[planeIndex];
            Vector3d normal = plane.get_normal();
            // Add the normal vector components to the triplet list
            constraintGradientTriplets.emplace_back(constraintCounter, 3 * surfaceVertexIndex, normal(0)); // Normal's x component
            constraintGradientTriplets.emplace_back(constraintCounter, 3 * surfaceVertexIndex + 1, normal(1)); // Normal's y component
            constraintGradientTriplets.emplace_back(constraintCounter, 3 * surfaceVertexIndex + 2, normal(2)); // Normal's z component
            constraintCounter++;    
            
        }
    }
    // Assert that the number of triplets matches the number of active constraints
    if (constraintCounter != m) {
        std::cerr << bcolors.FAIL << "Error: constraintCounter " << constraintCounter << " does not match the number of active constraints " << m << "in the plane contact gradient computation." << bcolors.ENDC << std::endl;
        assert(constraintCounter == m);
    }
    return constraintGradientTriplets;
}


template<int vertexDim, int elementDim>
bool PlaneContact<vertexDim, elementDim>::update_active_constraints(const VectorXd& vertices) {

    int numActiveConstraints = 0; // counter for active constraints
    // Loop over each surface vertex
    for (unsigned int i = 0; i < this->surfaceVerticesIdx_.size(); i++) {

        int surfaceVertexIndex = this->surfaceVerticesIdx_(i);
        // Extract vertex position
        Vector3d vertex = vertices.segment<3>(3 * surfaceVertexIndex);

        double minDistance = std::numeric_limits<double>::max();
        int closestPlaneIndex = -1;

        // Check the vertex against all planes, find the closest plane
        for (unsigned int planeIndex = 0; planeIndex < this->planes_.size(); ++planeIndex) {
            const Plane3D& plane = this->planes_[planeIndex];
            double distance = plane.distance_to_point(vertex);
            if (distance < PLANE_CONSTRAINT_THRESHOLD && distance < minDistance) {
                minDistance = distance;
                closestPlaneIndex = planeIndex;
            }
        }
        
        // Update the planeCollisionMask_ for the surface vertex (-1 if no collision, plane index otherwise)
        this->planeCollisionMask_[i] = closestPlaneIndex;
        if (closestPlaneIndex != -1) {
            numActiveConstraints++;
        }
    }

    // Update the number of active constraints
    this->set_num_active_constraints(numActiveConstraints);

    if (static_cast<Eigen::Index>(this->planeCollisionMask_.size()) != this->surfaceVerticesIdx_.size()) {
        return false; 
    }   
    return true; 
}

template class PlaneContact <3, 4>;
template class PlaneContact <3, 8>;
