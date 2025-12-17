#ifndef COMMON_H
#define COMMON_H

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <chrono>
#include <set>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <assert.h>
#include <memory>

using namespace Eigen;

// SIMULATION PARAMETERS

// Parameters for Newton's method and SQP method
// Maximum number of iterations for the Newton's method and SQP method
static const int MAX_ITER = 100; 
// Maximum number of iterations for the QP method
static const int MAX_ITER_QP = 250; 
// Tolerance for the Newton's method (dx < tol)
static constexpr double DX_TOL = 1e-9; 

// Parameters for constraint detection
// Tolerance for plane collision detection
static constexpr double PLANE_CONSTRAINT_THRESHOLD = 0.01; 
// Tolerance for spatial collision detection
static constexpr double SPATIAL_CONSTRAINT_THRESHOLD = 0.01;
// Tolerance for disk planar collision detection
static constexpr double DISK_PLANAR_CONSTRAINT_THRESHOLD = 0.01; 
// Tolerance for disk radial collision detection
static constexpr double DISK_RADIAL_CONSTRAINT_THRESHOLD = 0.001; 
// Threshold for checking perpendicular vectors with dot product
static constexpr double EPSILON_DP = 1e-6; 

// Colors from https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
struct {
    std::string HEADER      = "\033[95m";
    std::string OKBLUE      = "\033[94m";
    std::string OKCYAN      = "\033[96m";
    std::string OKGREEN     = "\033[92m";
    std::string WARNING     = "\033[93m";
    std::string FAIL        = "\033[91m";
    std::string ENDC        = "\033[0m";
    std::string BOLD        = "\033[1m";
    std::string UNDERLINE   = "\033[4m";
} bcolors;

// Visualization options for saving VTU files
enum class VisualType {VectorVertex, VectorElement, ScalarVertex, ScalarElement};
struct VisualOption {
    // Name of the dataset to be visualized
    std::string name;
    // Type of data: vector per vertex, vector per element, scalar per vertex, scalar per element
    VisualType type; 
    // Data matrix
    Eigen::MatrixXd data;
    // Threshold for visualization (values below threshold are set to zero)
    double threshold = - std::numeric_limits<double>::infinity(); 
};

// Define the OMP reduction for VectorXd
#pragma omp declare reduction (+: VectorXd: omp_out=omp_out+omp_in) initializer(omp_priv=VectorXd::Zero(omp_orig.size()))

// Define the OMP reduction for MatrixXd
#pragma omp declare reduction (+: MatrixXd: omp_out=omp_out+omp_in) initializer(omp_priv=MatrixXd::Zero(omp_orig.rows(), omp_orig.cols()))

// Macro to suppress unused variable warnings
#define UNUSED(x) (void)(x)

#endif
