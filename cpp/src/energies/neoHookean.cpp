#include "neoHookean.h"

template<int vertexDim, int elementDim>
NeoHookeanEnergy<vertexDim, elementDim>::NeoHookeanEnergy(
    double youngsModulus,
    double poissonsRatio,
    std::array<Matrix<double, vertexDim, vertexDim>, 9>& unitMatrices 
) : ElementEnergy<vertexDim, elementDim>(unitMatrices),
        youngsModulus_(youngsModulus), poissonsRatio_(poissonsRatio), 
        lambda_((youngsModulus * poissonsRatio) / ((1 + poissonsRatio) * (1 - 2 * poissonsRatio))),
        mu_(youngsModulus / (2 * (1 + poissonsRatio))){}


template<int vertexDim, int elementDim>
double NeoHookeanEnergy<vertexDim, elementDim>::compute_energy(
    const Matrix<double, elementDim, vertexDim>& vertices, 
    const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
    const MatrixXd& actuation) const {

    UNUSED(vertices); UNUSED(actuation);

    // Initialize Elastic energy (Neo-Hookean)
    double elasticEnergy = 0;

    // Normalized quadrature weights:
    int nquadpoints = deformationGradients.size();
    double quadratureWeights = 1. / nquadpoints; 

    // Compute integral over the element
    for (int s = 0; s < nquadpoints; ++s) {
        const Matrix<double, vertexDim, vertexDim>& F = deformationGradients[s];
        double J = F.determinant();
        double logJ = std::log(J);

        elasticEnergy += (0.5 * this->mu_ * ((F.transpose() * F).trace() - vertexDim)
                                - this->mu_ * logJ
                                + 0.5 * this->lambda_ * logJ * logJ) * quadratureWeights;
    }
    return elasticEnergy;
}


template<int vertexDim, int elementDim>
Vector<double, elementDim*vertexDim> NeoHookeanEnergy<vertexDim, elementDim>::compute_gradient(
    const Matrix<double, elementDim, vertexDim>& vertices, 
    const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
    const std::vector<Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>>& deformationHessians,
    const MatrixXd& actuation) const {

    UNUSED(vertices); UNUSED(actuation);

    // Elastic energy gradient (Neo-Hookean), dE/dx (1x12) = dE/dF (1x9) * dF/dx (9x12)
    Matrix<double, 1, elementDim*vertexDim> elasticGradient = Matrix<double, 1, elementDim*vertexDim>::Zero(); 

    // Normalized quadrature weights:
    int nquadpoints = deformationGradients.size();
    double quadratureWeights = 1. / nquadpoints; 

    // Compute integral over the element
    for (int s = 0; s < nquadpoints; ++s) {
        const Matrix<double, vertexDim, vertexDim>& F = deformationGradients[s];
        Matrix<double, vertexDim, vertexDim> FinvT = F.inverse().transpose();
        Matrix<double, 1, vertexDim * vertexDim> flattendF = F.transpose().reshaped();
        Matrix<double, 1, vertexDim * vertexDim> flattendFinvT = FinvT.transpose().reshaped();
        double J = F.determinant();
        double logJ = std::log(J);

        Matrix<double, 1, vertexDim*vertexDim> dEdF = this->mu_ * flattendF + (this->lambda_ * logJ - this->mu_) * flattendFinvT;
        elasticGradient += dEdF * deformationHessians[s] * quadratureWeights;
    }
    return elasticGradient;
}


template<int vertexDim, int elementDim>
Matrix<double, elementDim*vertexDim, elementDim*vertexDim> NeoHookeanEnergy<vertexDim, elementDim>::compute_hessian(
    const Matrix<double, elementDim, vertexDim>& vertices, 
    const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
    const std::vector<Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>>& deformationHessians,
    const MatrixXd& actuation) const {
    
    UNUSED(vertices); UNUSED(actuation);
    
    // Normalized quadrature weights:
    int nquadpoints = deformationGradients.size();
    double quadratureWeights = 1. / nquadpoints; 
    
    // Elastic energy hessian (12x12)
    Matrix<double, elementDim*vertexDim, elementDim*vertexDim> elasticHessian = Matrix<double, elementDim*vertexDim, elementDim*vertexDim>::Zero();

    // Compute integral over the element
    for (int s = 0; s <nquadpoints; ++s) {
        const Matrix<double, vertexDim, vertexDim>& F = deformationGradients[s];
        Matrix<double, vertexDim, vertexDim> FinvT = F.inverse().transpose();
        Matrix<double, 1, vertexDim * vertexDim> flattendFinvT = FinvT.transpose().reshaped();
       
        double J = F.determinant();
        if (J <= 0) {
            std::cout << bcolors.FAIL << "Inverted Elements. Please decrease timestep or use adaptive timestepping!" << bcolors.ENDC << std::endl;
            throw std::runtime_error("Inverted Elements");
        }
        double logJ = std::log(J);

        // We first compute the gradient of FinvT (FinvTGrad). Since FinvT is 3x3, FinvTGrad is 9x9
        Matrix<double, vertexDim*vertexDim, vertexDim*vertexDim> FinvTGrad; 
        
        // Calculate -FinvT * E_i^T * FinvT for each E_i, then flatten the resulting matrix and put as a column in FinvTGrad
        for (std::size_t i = 0; i < this->unitMatrices_.size(); ++i) {
            Matrix<double, vertexDim, vertexDim> blockMatrix = - FinvT * this->unitMatrices_[i].transpose() * FinvT;
            FinvTGrad.col(i) =  blockMatrix.transpose().reshaped();
        }

        if (logJ != logJ) {
            std::cout << "logJ is NaN." << std::endl;
            std::cout << "J: " << J << std::endl;
            std::cout << "logJ: " << logJ << std::endl;
        }
        if ((FinvTGrad.array() == FinvTGrad.array()).all() == false) {
            std::cout << "FinvTGrad contains NaN values." << std::endl;
        }

        elasticHessian += 
            (((this->mu_ * Matrix<double, vertexDim*vertexDim, vertexDim*vertexDim>::Identity(vertexDim*vertexDim, vertexDim*vertexDim) +
            this->lambda_ * flattendFinvT.transpose() * flattendFinvT + // FlattenedFinvT is a row vector (1x9), hence the results 9x9
            (this->lambda_ * logJ - this->mu_) * FinvTGrad).transpose()) *deformationHessians[s]).transpose() * deformationHessians[s] * quadratureWeights;
    }
    return elasticHessian;
}

template class NeoHookeanEnergy <3, 4>;
template class NeoHookeanEnergy <3, 8>;
