#ifndef PLANE3D_H
#define PLANE3D_H

#include "common.h"

/**
 * @class Plane3D
 * @brief Geometric representation of a 3D plane defined by a point and a normal.
 *
 * Provides routines for point–plane queries (distance, projection, on-plane test)
 * as well as generating a visualization quad centered on the plane. Used by the
 * contact and collision system for detecting and resolving plane-based contacts.
 */
class Plane3D {
public:
    // Default constructor
    Plane3D() : point_(Vector3d(0.0, 0.0, 0.0)), normal_(Vector3d(0.0, 0.0, 1.0)) {}

    // Constructor to initialize the plane with a point and a normal vector
    Plane3D(Vector3d point, Vector3d normal) 
        : point_(point), normal_(normal.normalized()) {squarePoints_ = create_square_points(1.0);}

    // Function to check if a given point lies on the plane
    bool is_point_on_plane(Vector3d otherPoint) const {
        Vector3d vec = otherPoint - point_;
        return std::abs(vec.dot(normal_)) < EPSILON_DP;  // Check if dot product is close to zero
    }
    
    // Function to calculate the perpendicular distance from a point to the plane
    double distance_to_point(Vector3d otherPoint)  const {
        Vector3d vec = otherPoint - point_;
        return vec.dot(normal_);
    }

    // Function to project a point onto the plane
    Vector3d project_point_on_plane(Vector3d otherPoint) const {
        double distance = distance_to_point(otherPoint);
        return otherPoint - distance * normal_;
    }

    // Create vector with 4 points, defining a 1m x 1m square on the plane with the point_ as the center
    std::vector<double> create_square_points(double squareSide) {
        // Create a 1m by 1m rectangle with the plane as the normal and the point as the center
        Vector3d x = Vector3d::UnitX();
        Vector3d y = Vector3d::UnitY();
        Vector3d u = x.cross(normal_);
        if (normal_.isApprox(x)) {
            u = y.cross(normal_);
        } 
        Vector3d v = normal_.cross(u);
        u.normalize();
        v.normalize();
        Vector3d p1 = point_ + 0.5*squareSide * u + 0.5*squareSide * v;
        Vector3d p2 = point_ - 0.5*squareSide * u + 0.5*squareSide * v;
        Vector3d p3 = point_ - 0.5*squareSide * u - 0.5*squareSide * v;
        Vector3d p4 = point_ + 0.5*squareSide * u - 0.5*squareSide * v;

        std::vector<double> squarePointsFlattened = {p1(0), p1(1), p1(2), p2(0), p2(1), p2(2), p3(0), p3(1), p3(2), p4(0), p4(1), p4(2)};
        return squarePointsFlattened;
    }

    // Getter for point_
    Vector3d get_point() const { return point_; }

    // Setter for point_
    void set_point(Vector3d point) { 
        point_ = point; 
        squarePoints_ = create_square_points(1.0);}

    // Getter for normal_
    Vector3d get_normal() const { return normal_; }

    // Setter for normal_
    void set_normal(Vector3d normal) { 
        normal_ = normal; 
        squarePoints_ = create_square_points(1.0);
    }

    // Getter for squarePoints_
    std::vector<double> get_square_points() const { return squarePoints_; }

private:
    // Point on the plane
    Vector3d point_;
    // Normal vector of the plane
    Vector3d normal_;
    // Vector with 4 points, defining a 1m x 1m square on the plane with the point_ as the center for visualization
    std::vector<double> squarePoints_;
};

#endif