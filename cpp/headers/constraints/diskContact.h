#ifndef DISKCONTACT_H
#define DISKCONTACT_H

#include "constraint.h"
#include "disk3D.h"

/**
 * @class DiskContact
 * @brief Inequality contact constraint between a surface mesh and one or more disks.
 *
 * Inherits from Constraint. Activates a surface vertex when it lies sufficiently
 * close to a disk plane and within the disk's radial footprint. Once any vertex
 * touches a disk, a “captured” footprint of nearby vertices is frozen to prevent 
 * contact flickering. The footprint is released only when all captured vertices 
 * rise clearly above the disk again. Active constraints behave like plane-contact 
 * constraints defined by each disk's plane.
 */
template<int vertexDim, int elementDim>
class DiskContact : public Constraint<vertexDim, elementDim> {

public:
    /**
     * @brief Construct a disk contact constraint.
     *
     * Initializes contact handling over a set of disks and a list of surface
     * vertices. All surface vertices start as inactive. Plane and radial thresholds
     * determine how close a vertex must be to be considered a collision candidate.
     *
     * @param numVertices        Total number of mesh vertices.
     * @param disks              List of disk obstacles.
     * @param surfaceVerticesIdx Indices of vertices forming the object's surface.
     * @param planeThreshold     Allowed proximity to disk plane before activation.
     * @param radialThreshold    Extra radial tolerance beyond disk radius.
     */
    DiskContact(int numVertices, 
                std::vector<Disk3D> disks, 
                VectorXi surfaceVerticesIdx,
                double planeThreshold = DISK_PLANAR_CONSTRAINT_THRESHOLD,
                double radialThreshold = DISK_RADIAL_CONSTRAINT_THRESHOLD)
    : Constraint<vertexDim, elementDim>(numVertices), 
      disks_(disks), 
      surfaceVerticesIdx_(surfaceVerticesIdx),
      diskCollisionMask_(std::vector<int>(surfaceVerticesIdx_.size(), -1)),
      planeThreshold_(planeThreshold),
      radialThreshold_(radialThreshold) {}
    
    // Computes the scalar constraint function 
    VectorXd compute_constraint(const VectorXd& vertices, const VectorXd& verticesPrev, double dt) const override;

    // Computes the gradient of the scalar constraint function 
    std::vector<Triplet<double>> compute_constraint_gradient(const VectorXd& vertices, const VectorXd& verticesPrev, double dt) const override;

    // Updates the active constraints 
    bool update_active_constraints(const VectorXd& vertices) override;

    // Returns the type of constraint: "equality" or "inequality"
    std::string get_constraint_type() const override {
        return "inequality";
    }

    // Returns the name of the constraint
    std::string get_constraint_name() const override {
        return "diskContact";
    }

    // Sets disks (input is a vector of Disk3D objects)
    void set_disks(const std::vector<Disk3D>& disks) {
        disks_ = disks;
    }

    // Sets a specific disk (input is center, normal, radius, and index of disk to set)
    void set_disk(const VectorXd& center, const VectorXd& normal, double radius, int diskIdx=0) {
        if(diskIdx < 0 || diskIdx >= static_cast<int>(disks_.size())) {
            std::cerr << "Error: diskIdx out of range in set_disk()." << std::endl;
            return;
        }
        disks_[diskIdx].set_center(center);
        disks_[diskIdx].set_normal(normal);
        disks_[diskIdx].set_radius(radius);
    }

protected:
    // Collection of disks
    std::vector<Disk3D> disks_;
    // Vector of surface vertices indices of object
    VectorXi surfaceVerticesIdx_;
    // Vector size of surfaceVerticesIdx_ that stores -1 if no collision, disk index otherwise for each surface vertex
    std::vector<int> diskCollisionMask_; 
    // If true, the active set of constraints is not updated 
    bool frozenActiveSet_ = false; 
    // Threshold distance to consider a vertex in contact with the disk plane
    double planeThreshold_;  
    // Threshold radial distance to consider a vertex in contact with the disk
    double radialThreshold_;  
};

#endif