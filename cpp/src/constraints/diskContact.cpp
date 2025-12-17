#include "diskContact.h"

template<int vertexDim, int elementDim>
VectorXd DiskContact<vertexDim, elementDim>::compute_constraint(const VectorXd& vertices, const VectorXd& verticesPrev, double dt) const {
    
    UNUSED(verticesPrev); UNUSED(dt); 

    // Number of active constraints 
    int m = this->get_num_active_constraints();
    VectorXd constraintVector(m); 

    // Only loop over surface vertices
    int constraintCounter = 0;
    for (unsigned int i = 0; i < this->surfaceVerticesIdx_.size(); i++) {
        
        // If no collision, skip this vertex
        int diskIndex = this->diskCollisionMask_[i];
        if (diskIndex == -1) { 
            continue;
        } else {
            // Get the index of the surface vertex
            int surfaceVertexIndex = this->surfaceVerticesIdx_(i);
            // Extract vertex position
            Vector3d vertex = vertices.segment<3>(3*surfaceVertexIndex);
            // Assert if the disk index is valid (it should be less than the number of disk and greater than or equal to 0)
            if (!(diskIndex < int(disks_.size()) && diskIndex >= 0)) {
                std::cerr << bcolors.FAIL << "Error: diskIndex " << diskIndex << " is out of bounds. It must be between 0 and " << disks_.size() - 1 << "." << bcolors.ENDC << std::endl;
                assert(diskIndex < int(disks_.size()) && diskIndex >= 0);
            }
            const Disk3D& disk = disks_[diskIndex];
            const Vector3d normal = disk.get_normal();
            const Vector3d center = disk.get_center();
            // Compute the constraint function 
            constraintVector(constraintCounter) = normal.dot(vertex - center);
            constraintCounter++;
        }   
    }
    // Assert that the constraint vector has the correct size
    if (constraintCounter != m) {
        std::cerr << bcolors.FAIL << "Error: constraintCounter " << constraintCounter << " does not match the number of active constraints " << m << " in the disk contact constraint computation." << bcolors.ENDC << std::endl;
        assert(constraintCounter == m);
    }
    return constraintVector;
}


template<int vertexDim, int elementDim>
std::vector<Triplet<double>> DiskContact<vertexDim, elementDim>::compute_constraint_gradient(const VectorXd& vertices, const VectorXd& verticesPrev, double dt) const {

    UNUSED(vertices); UNUSED(verticesPrev); UNUSED(dt); 

    // Number of active constraints 
    int m = this->get_num_active_constraints(); 
    std::vector<Triplet<double>> constraintGradientTriplets; // Triplet list for sparse matrix (m x 3n)
    constraintGradientTriplets.reserve(3 * size_t(m)); // Reserve space for triplets (3 entries per active constraint)
   
    // Loop over all surface vertices
    int constraintCounter = 0;
    for (unsigned int i = 0; i < this->surfaceVerticesIdx_.size(); i++) {
        // If no collision, skip this vertex
        int diskIndex = this->diskCollisionMask_[i];
        if (diskIndex == -1) { 
            continue;
        } else {
            int surfaceVertexIndex = this->surfaceVerticesIdx_(i);
            // Assert if the disk index is valid (it should be less than the number of disks and greater than or equal to 0)
            if (!(diskIndex < int(disks_.size()) && diskIndex >= 0)) {
                std::cerr << bcolors.FAIL << "Error: diskIndex " << diskIndex << " is out of bounds. It must be between 0 and " << disks_.size() - 1 << "." << bcolors.ENDC << std::endl;
                assert(diskIndex < int(disks_.size()) && diskIndex >= 0);
            }
            const Disk3D& disk = disks_[diskIndex];
            const Vector3d normal = disk.get_normal();
            // Add the normal vector components to the triplet list
            constraintGradientTriplets.emplace_back(constraintCounter, 3 * surfaceVertexIndex, normal(0)); // Normal's x component
            constraintGradientTriplets.emplace_back(constraintCounter, 3 * surfaceVertexIndex + 1, normal(1)); // Normal's y component
            constraintGradientTriplets.emplace_back(constraintCounter, 3 * surfaceVertexIndex + 2, normal(2)); // Normal's z component
            constraintCounter++;    
            
        }
    }
    // Assert that the number of triplets matches the number of active constraints
    if (constraintCounter != m) {
        std::cerr << bcolors.FAIL << "Error: constraintCounter " << constraintCounter << " does not match the number of active constraints " << m << "in the disk contact gradient computation." << bcolors.ENDC << std::endl;
        assert(constraintCounter == m);
    }
    return constraintGradientTriplets;
}


template<int vertexDim, int elementDim>
bool DiskContact<vertexDim, elementDim>::update_active_constraints(const VectorXd& vertices) {

    // Thresholds
    constexpr double clearFactor        = 3.0; // Unfreeze when clearly above (>= 3x planeThreshold_)
    constexpr double captureDepthFactor = 5.0; // Go this many thresholds "down" into object at first contact
    const double planeClearThreshold = clearFactor * planeThreshold_;
    const double captureDepth        = captureDepthFactor * planeThreshold_; // Allow negative plane distances down to -captureDepth

    // 1) Frozen: keep until all frozen verts are clearly above their disk ===
    if (frozenActiveSet_) {
        bool all_clear = true;
        // Iterate only over vertices that are part of the frozen set (mask != -1)
        for (size_t localIdx = 0; localIdx < diskCollisionMask_.size() && all_clear; ++localIdx) {
            const int diskIdx = diskCollisionMask_[localIdx];
            if (diskIdx == -1) continue; // not in frozen set
            const int vIdx = surfaceVerticesIdx_(static_cast<int>(localIdx));
            const Vector3d x = vertices.segment<3>(3 * vIdx);
            // "Clearly above" = distance to the (same) disk plane >= planeClearThreshold
            const Disk3D& disk = disks_.at(static_cast<size_t>(diskIdx));
            const double planeDistance = disk.signed_distance_to_plane(x);
            if (planeDistance < planeClearThreshold) {
                all_clear = false; // Still near/under the plane → stay frozen
            }
        }
        // Auto-unfreeze: disk is above the whole previously-colliding footprint
        if (all_clear) {
            frozenActiveSet_ = false;
            std::fill(diskCollisionMask_.begin(), diskCollisionMask_.end(), -1);
            this->set_num_active_constraints(0);
        } else {
            // Stay frozen; keep current mask and just recompute count
            int cnt = 0;
            for (int d : diskCollisionMask_) if (d != -1) ++cnt;
            this->set_num_active_constraints(cnt);
        }
        return true;
    }
    // 2) Not frozen: detect any hit using your original near-plane + radial tests ===
    // First, see which disks have any candidate under the standard thresholds.
    std::vector<bool> diskHasHit(disks_.size(), false);
    {
        for (int i = 0; i < surfaceVerticesIdx_.size(); ++i) {
            const int vIdx = surfaceVerticesIdx_(i);
            const Vector3d x = vertices.segment<3>(3 * vIdx);
            for (size_t d = 0; d < disks_.size(); ++d) {
                const Disk3D& disk = disks_[d];
                const double planeDistance  = disk.signed_distance_to_plane(x);
                if (planeDistance >= planeThreshold_) continue;
                const double radialDistance = disk.radial_distance_on_plane(x);
                if (radialDistance > disk.get_radius() + radialThreshold_) continue;
                diskHasHit[d] = true; // Mark this disk as contacting
            }
        }
    }
    // If no disk has any candidate, stay unfrozen and clear mask
    bool anyHit = false; 
    for (bool b : diskHasHit) {if (b) {anyHit = true; break;}}
    if (!anyHit) {
        std::fill(diskCollisionMask_.begin(), diskCollisionMask_.end(), -1);
        this->set_num_active_constraints(0);
        return true;
    }

    // 3) First contact: CAPTURE footprint by projecting onto the disk plane
    // For each disk that had a hit, include ALL surface verts whose projection is within radius
    // and whose signed plane distance is within [-captureDepth, +planeThreshold_].
    std::vector<int> newDiskCollisionMask(surfaceVerticesIdx_.size(), -1);
    int numActive = 0;

    for (size_t d = 0; d < disks_.size(); ++d) {
        if (!diskHasHit[d]) continue; // Only freeze disks that actually contacted this step
        const Disk3D& disk = disks_[d];

        for (int i = 0; i < surfaceVerticesIdx_.size(); ++i) {
            const int vIdx = surfaceVerticesIdx_(i);
            const Vector3d x = vertices.segment<3>(3 * vIdx);

            // Plane window allows going below the plane by captureDepth
            const double planeDistance  = disk.signed_distance_to_plane(x);
            if (planeDistance >  planeThreshold_)   continue;  // Too far above
            if (planeDistance < -captureDepth)      continue;  // Too deep below

            // Radial test on orthogonal projection
            const double radialDistance = disk.radial_distance_on_plane(x);
            if (radialDistance > disk.get_radius() + radialThreshold_) continue;

            // Mark this surface vertex as colliding with disk d
            // Note: if multiple disks overlap (rare), prefer the one with smaller |planeDistance|
            int& slot = newDiskCollisionMask[static_cast<size_t>(i)];
            if (slot == -1) {
                slot = static_cast<int>(d);
                ++numActive;
            } else {
                // Optional tie-break: keep the closer plane
                const Disk3D& prev = disks_[static_cast<size_t>(slot)];
                const double prevPlane = prev.signed_distance_to_plane(x);
                if (std::abs(planeDistance) < std::abs(prevPlane)) {
                    // Reassignment doesn't change numActive count
                    slot = static_cast<int>(d);
                }
            }
        }
    }
    // If, somehow, still nothing captured, keep unfrozen
    if (numActive == 0) {
        std::fill(diskCollisionMask_.begin(), diskCollisionMask_.end(), -1);
        this->set_num_active_constraints(0);
        return true;
    }
    
    // 4) Freeze the captured footprint, eliminate flicker ===
    diskCollisionMask_.swap(newDiskCollisionMask);   
    this->set_num_active_constraints(numActive);
    frozenActiveSet_ = true;
    return true;
}

template class DiskContact <3, 4>;
template class DiskContact <3, 8>;