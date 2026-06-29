#ifndef SOFTCONMUSCLEENERGY_H
#define SOFTCONMUSCLEENERGY_H

#include "elementEnergy.h"
#include "params.h"

/**
 * @class SoftconMuscleEnergy
 * @brief SoftCon-style contractile muscle energy for embedded muscle fibers.
 *
 * Implements the muscle energy, gradient, and Hessian used in SoftCon
 * (directional fiber contraction with scalar activation) and inherits
 * from ElementEnergy. Based on 
 * https://mrl.snu.ac.kr/publications/ProjectSoftCon/SoftCon.html
 */
template <int vertexDim, int elementDim>
class SoftconMuscleEnergy : public ElementEnergy<vertexDim, elementDim>
{
public:
    /**
     * @brief Construct a SoftCon muscle energy term for a given muscle group.
     *
     * Initializes muscle index, stiffness, and fiber direction and enables actuation for this energy.
     */
    SoftconMuscleEnergy(
        int muscleGroup,
        double muscleStiffness,
        VectorXd muscleDirection,
        std::array<Matrix<double, vertexDim, vertexDim>, 9>& unitMatrices,
        const std::vector<Matrix<double, vertexDim * vertexDim, vertexDim * elementDim>>& deformationHessians);

    // Compute softcon muscle energy
    double compute_energy(
        const Matrix<double, elementDim, vertexDim>& vertices, 
        const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
        const MatrixXd& actuation) const override;

    // Compute gradient of softcon muscle energy
    Vector<double, elementDim*vertexDim> compute_gradient(
        const Matrix<double, elementDim, vertexDim>& vertices, 
        const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
        const std::vector<Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>>& deformationHessians,
        const MatrixXd& actuation) const override;

    // Compute hessian of softcon muscle energy 
    Matrix<double, elementDim*vertexDim, elementDim*vertexDim> compute_hessian(
        const Matrix<double, elementDim, vertexDim>& vertices, 
        const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
        const std::vector<Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>>& deformationHessians,
        const MatrixXd& actuation) const override;

    // Returns the name of the energy
    std::string get_energy_name() const override {
        return "softconMuscle";
    }

private:
    // Indexing into actuation vector, to identify how much activation to apply to each muscle tetrahedron.
    const int muscleGroup_;                     
    // Muscle stiffness  
    const double k_;                               
    // Muscle fibre direction
    const Vector3d m_;                             
    // Constant directed Hessian term used by the SoftCon model.
    std::vector<Matrix<double, elementDim * vertexDim, elementDim * vertexDim>> directedDeformationHessian_;

    /**
     * @brief Class to manage time-varying muscle activation patterns.
     * 
     * Provides methods to retrieve activation levels based on predefined
     * temporal patterns, track activation progress, and modify activation
     * signals during simulation.
     */
    class Activation
    {
    public:
        // Getters and setters
        double get_activation(double forward_in_time = 0) const;
        double get_tracker() const { return activationTracker_; };
        void reset_tracker() { activationTracker_ = 0; };
        void update_activation_progress(const double dt) { activationTracker_ += dt; };
        // Potential use in the future to change activation signal within simulation
        void set_activation_pattern(std::string pattern) { currentActivationPattern_ = pattern; };

    private:
        // Track current progress in the activation
        double activationTracker_; 
        // Current activation pattern
        std::string currentActivationPattern_;
    };
    
    // Muscle activation instance
    Activation act;
};

#endif
