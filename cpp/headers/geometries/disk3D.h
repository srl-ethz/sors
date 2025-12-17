#ifndef DISK3D_H
#define DISK3D_H

#include "common.h"

/**
 * @class Disk3D
 * @brief Simple geometric representation of a circular disk in 3D space.
 *
 * Defines a planar circular cap via center position, unit normal direction,
 * and radius. Provides helper routines for computing signed distance to the
 * supporting plane and radial in-plane distance, used in contact and proximity
 * queries.
 */
class Disk3D {
public:
    // Default constructor (x-plane)
    Disk3D() : center_(Vector3d::Zero()), normal_(Vector3d::UnitZ()), radius_(1.0) {}

    // Constructor to initialize the disk with a center, a normal vector and a radius
    Disk3D(const Vector3d& center, const Vector3d& normal, double radius)
        : center_(center), normal_(normal.normalized()), radius_(radius) {}

    // Getters
    inline const Vector3d& get_center() const { return center_; }
    inline const Vector3d& get_normal() const { return normal_; }
    inline double get_radius() const { return radius_; }

    // Setters
    inline void set_center(const Vector3d& center) { center_ = center; }
    inline void set_normal(const Vector3d& normal) { normal_ = normal.normalized(); }
    inline void set_radius(double radius) { radius_ = radius; }

    // Signed distance to disk's plane (positive if on normal side)
    inline double signed_distance_to_plane(const Vector3d& p) const {
        return normal_.dot(p - center_);
    }

    // Radial distance from the disk center projected onto the plane
    inline double radial_distance_on_plane(const Vector3d& p) const {
        // Remove normal component -> projection onto plane
        Vector3d v = p - center_;
        Vector3d tangential = v - normal_.dot(v) * normal_;
        return tangential.norm();
    }

private:
    // Center of the disk
    Vector3d center_;
    // Normal of the disk
    Vector3d normal_;
    // Radius of the disk
    double radius_;
};

#endif