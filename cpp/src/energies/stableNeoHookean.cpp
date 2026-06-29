#include "stableNeoHookean.h"

template<int vertexDim, int elementDim>
StableNeoHookeanEnergy<vertexDim, elementDim>::StableNeoHookeanEnergy(
    double youngsModulus,
    double poissonsRatio,
    std::array<Matrix<double, vertexDim, vertexDim>, 9>& unitMatrices 
) : ElementEnergy<vertexDim, elementDim>(unitMatrices),
        youngsModulus_(youngsModulus), poissonsRatio_(poissonsRatio), 
        // Stable Neo-Hookean parameters reparameterizing the Lame Parameters
        mu_((4.0/3.0)*(youngsModulus / (2 * (1 + poissonsRatio)))),
        lambda_(((youngsModulus * poissonsRatio) / ((1 + poissonsRatio) * (1 - 2 * poissonsRatio))) + ((5.0/6.0)*mu_)),
        alpha_(1 + (3.0/4.0)*(mu_/lambda_)) {}


template<int vertexDim, int elementDim>
double StableNeoHookeanEnergy<vertexDim, elementDim>::compute_energy(
    const Matrix<double, elementDim, vertexDim>& vertices, 
    const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
    const MatrixXd& actuation) const {

    UNUSED(vertices); UNUSED(actuation);

    // Stable elastic energy (Neo-Hookean)
    double stableElasticEnergy = 0;
    
    // Normalized quadrature weights:
    int nquadpoints = deformationGradients.size();
    double quadratureWeights = 1. / nquadpoints; 
    
    // Compute the integral over the element
    for (int s = 0; s <nquadpoints; ++s) {
        const Matrix<double, vertexDim, vertexDim>& F = deformationGradients[s];
        double J = F.determinant();
        Matrix<double, vertexDim, vertexDim> FTF = F.transpose() * F;
        double I_C = FTF.trace();

        // Compute function terms
        double term1 = (this->mu_ / 2.0) * (I_C - 3.0);                // mu / 2 * (I_C - 3)
        double term2 = (this->lambda_ / 2.0) * (J - this->alpha_)*(J - this->alpha_); // lambda / 2 * (J - alpha)^2
        double term3 = -(this->mu_ / 2.0) * std::log(I_C + 1.0);       // -mu / 2 * log(I_C + 1)

        stableElasticEnergy += (term1 + term2 + term3) * quadratureWeights;
    }
    return stableElasticEnergy;
}


template<int vertexDim, int elementDim>
Vector<double, elementDim*vertexDim> StableNeoHookeanEnergy<vertexDim, elementDim>::compute_gradient(
    const Matrix<double, elementDim, vertexDim>& vertices, 
    const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
    const std::vector<Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>>& deformationHessians,
    const MatrixXd& actuation) const {

    UNUSED(vertices); UNUSED(actuation);

    // Stable elastic gradient (Neo-Hookean)
    Matrix<double, 1, elementDim*vertexDim> gradient = Matrix<double, 1, elementDim*vertexDim>::Zero();
    
    // Normalized quadrature weights:
    int nquadpoints = deformationGradients.size();
    double quadratureWeights = 1. / nquadpoints; 
    
    // Compute the integral over the element
    for (int s = 0; s < nquadpoints; ++s) {
        const Matrix<double, vertexDim, vertexDim>& F = deformationGradients[s];
        Matrix<double, 1, vertexDim * vertexDim> flattendF = F.transpose().reshaped();
        double J = F.determinant();
        Matrix<double, vertexDim, vertexDim> FTF = F.transpose() * F;
        double I_C = FTF.trace();
        Matrix<double, vertexDim, vertexDim> FinvT = F.inverse().transpose();
        Matrix<double, 1, vertexDim * vertexDim> flattendFinvT = FinvT.transpose().reshaped();

        // Compute individual gradient terms (flattened rowwise, 1x9)
        Matrix<double, 1, vertexDim * vertexDim> term1 = this->mu_ * flattendF;  // mu * F
        Matrix<double, 1, vertexDim * vertexDim> term2 = this->lambda_ * (J - this->alpha_) * J * flattendFinvT;  // lambda * (J - alpha) * J * F^(-T)
        Matrix<double, 1, vertexDim * vertexDim> term3 = -(this->mu_ * flattendF) / (I_C + 1.0);  // -(mu * F) / (I_C + 1)

        // Combine terms to get the gradient
        Matrix<double, 1, vertexDim * vertexDim> dPsidF = term1 + term2 + term3; // 1x9 matrix

        // Compute the gradient of the energy
        gradient += dPsidF * deformationHessians[s] * quadratureWeights;
    }
    return gradient;
}


template<int vertexDim, int elementDim>
Matrix<double, elementDim*vertexDim, elementDim*vertexDim> StableNeoHookeanEnergy<vertexDim, elementDim>::compute_hessian(
    const Matrix<double, elementDim, vertexDim>& vertices, 
    const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
    const std::vector<Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>>& deformationHessians,
    const MatrixXd& actuation) const {
    
    UNUSED(vertices); UNUSED(actuation);
    
    // Normalized quadrature weights:
    int nquadpoints = deformationGradients.size();
    double quadratureWeights = 1. / nquadpoints; 

    // Elastic energy hessian (12x12)                          
    Eigen::Matrix<double, elementDim*vertexDim, elementDim*vertexDim> hessian = Eigen::Matrix<double, elementDim*vertexDim, elementDim*vertexDim>::Zero();

    // Compute the integral over the element
    for (int s = 0; s < nquadpoints; ++s) {
        const Matrix<double, vertexDim, vertexDim>& F = deformationGradients[s];
        Matrix<double, 1, vertexDim * vertexDim> flattendF = F.transpose().reshaped();
        double J = F.determinant();
        Matrix<double, vertexDim, vertexDim> FTF = F.transpose() * F;
        double I_C = FTF.trace();
        Matrix<double, vertexDim, vertexDim> FinvT = F.inverse().transpose();
        Matrix<double, 1, vertexDim * vertexDim> flattendFinvT = FinvT.transpose().reshaped();
        Matrix<double, vertexDim*vertexDim, vertexDim*vertexDim> identity9 = Matrix<double, vertexDim*vertexDim, vertexDim*vertexDim>::Identity(vertexDim*vertexDim, vertexDim*vertexDim);

        // We first compute the gradient of FinvT (FinvTGrad). Since FinvT is 3x3, FinvTGrad is 9x9
        Matrix<double, vertexDim*vertexDim, vertexDim*vertexDim> FinvTGrad; 
        // Calculate -FinvT * E_i^T * FinvT for each E_i, then flatten the resulting matrix and put as a column in FinvTGrad
        for (std::size_t i = 0; i < this->unitMatrices_.size(); ++i) {
            Matrix<double, vertexDim, vertexDim> blockMatrix = - FinvT * this->unitMatrices_[i].transpose() * FinvT;
            FinvTGrad.col(i) =  blockMatrix.transpose().reshaped();
        }

        // --- Term 1: mu * I ---
        Eigen::Matrix<double, vertexDim * vertexDim, vertexDim * vertexDim> term1 = this->mu_ * identity9;

        // --- Term 2: (2J - alpha) * lambda * J * (F^(-T) ⊗ F^(-T)) ---
        Eigen::Matrix<double, vertexDim * vertexDim, vertexDim * vertexDim> FinvT_outer = flattendFinvT.transpose() * flattendFinvT;  // Outer product, flattenedFinvT is a row vector (1x9), hence the results 9x9
        Eigen::Matrix<double, vertexDim * vertexDim, vertexDim * vertexDim> term2 = (2 * J - this->alpha_) * this->lambda_ * J * FinvT_outer;

        // --- Term 3: lambda * (J - alpha) * J * M, here M = FinvTGrad ---
        Eigen::Matrix<double, vertexDim * vertexDim, vertexDim * vertexDim> term3 = this->lambda_ * (J - this->alpha_) * J * FinvTGrad;

        // --- Term 4: ((-mu * I) / (I_C + 1)) + (2 * mu * (F ⊗ F)) / (I_C + 1)^2 ---
        Eigen::Matrix<double, vertexDim * vertexDim, vertexDim * vertexDim> F_outer = flattendF.transpose() * flattendF;  // Outer product, flattenedF is a row vector (1x9), hence the results 9x9
        Eigen::Matrix<double, vertexDim * vertexDim, vertexDim * vertexDim> term4 = ((-this->mu_ * identity9) / (I_C + 1.0)) + ((2 * this->mu_ * F_outer) / ((I_C + 1.0)*(I_C + 1.0)));

        // A = Term 1 + Term 2 + Term 3 + Term 4
        Eigen::Matrix<double, vertexDim * vertexDim, vertexDim * vertexDim> A = term1 + term2 + term3 + term4;

        // Compute the hessian of the energy: H = (A^T * nablaF)^T * nablaF where nablaF is the deformation hessian
        hessian += (A.transpose() * deformationHessians[s]).transpose() * deformationHessians[s] * quadratureWeights;

        // if (J <= 0) {
        // std::cout << bcolors.OKCYAN << "Inverted Elements in stable NeoHookean!" << bcolors.ENDC << std::endl;
        // }
    }
    return hessian;
}

template class StableNeoHookeanEnergy <3, 8>;
template class StableNeoHookeanEnergy <3, 4>;
