#include "pseudostrain.h"

template<int vertexDim, int elementDim>
PseudostrainEnergy<vertexDim, elementDim>::PseudostrainEnergy(
    double b, double c, double D,
    std::array<Matrix<double, vertexDim, vertexDim>, 9>& unitMatrices 
) : ElementEnergy<vertexDim, elementDim>(unitMatrices),
    b_(b), c_(c), D_(D){}


template<int vertexDim, int elementDim>
double PseudostrainEnergy<vertexDim, elementDim>::compute_energy(
    const Matrix<double, elementDim, vertexDim>& vertices, 
    const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
    const MatrixXd& actuation) const {

    UNUSED(vertices); UNUSED(actuation);

    // Elastic energy 
    double elasticEnergy = 0;

    // Normalized quadrature weights:
    int nquadpoints = deformationGradients.size();
    double quadratureWeights = 1. / nquadpoints; 
    
    // Compute integral over the element
    for (int s = 0; s < nquadpoints; ++s) {
        const Matrix<double, vertexDim, vertexDim>& F = deformationGradients[s];
        elasticEnergy += (compute_deviatoric_energy(F) + compute_volume_energy(F)) * quadratureWeights;
    }
    return elasticEnergy;
}


template<int vertexDim, int elementDim>
double PseudostrainEnergy<vertexDim, elementDim>::compute_deviatoric_energy(
    const Matrix<double, vertexDim, vertexDim>& deformationGradient) const {
    // Deviatoric part
    const Matrix<double, vertexDim, vertexDim>& F = deformationGradient;
    double J = F.determinant();
    double I_bar = (std::pow(J, -2/3.) * F.transpose() * F).trace();
    double deviatoricEnergy = c_*(std::exp(b_*(I_bar-3))-1);
    return deviatoricEnergy;
}


template<int vertexDim, int elementDim>
double PseudostrainEnergy<vertexDim, elementDim>::compute_volume_energy(
    const Matrix<double, vertexDim, vertexDim>& deformationGradient) const {
    // Volumetric part
    const Matrix<double, vertexDim, vertexDim>& F = deformationGradient;
    double J = F.determinant();
    double volumeEnergy = 1/D_*(J-1)*(J-1);
    return volumeEnergy;
}


template<int vertexDim, int elementDim>
Vector<double, elementDim*vertexDim> PseudostrainEnergy<vertexDim, elementDim>::compute_gradient(
    const Matrix<double, elementDim, vertexDim>& vertices, 
    const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
    const std::vector<Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>>& deformationHessians,
    const MatrixXd& actuation) const {

    UNUSED(vertices); UNUSED(actuation);

    Vector<double, elementDim*vertexDim> grad = Vector<double, elementDim*vertexDim>::Zero();

    // Normalized quadrature weights:
    int nquadpoints = deformationGradients.size();
    double quadratureWeights = 1. / nquadpoints; 

    // Compute integral over the element
    for (int s = 0; s < nquadpoints; ++s) {
        const auto& F = deformationGradients[s];
        const auto& nablaF = deformationHessians[s];
        grad += (compute_deviatoric_gradient(F, nablaF) + compute_volume_gradient(F, nablaF)) * quadratureWeights;
    }
    return grad;
}


template<int vertexDim, int elementDim>
Vector<double, elementDim*vertexDim> PseudostrainEnergy<vertexDim, elementDim>::compute_deviatoric_gradient(
    const Matrix<double, vertexDim, vertexDim>& deformationGradient, const Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>& deformationHessian) const {
        
    const Matrix<double, vertexDim, vertexDim>& F = deformationGradient;
    Matrix<double, vertexDim, vertexDim> FinvT = F.inverse().transpose();

    double J = F.determinant();
    double I_bar = (std::pow(J, -2/3.) * F.transpose() * F).trace();

    Matrix<double, 1, vertexDim * vertexDim> deviatoric_gradientdF = b_ * c_ * std::exp(b_*(I_bar-3)) * \
        (-2/3.*std::pow(J, -2/3.)*((F.transpose() * F).trace())* FinvT + 2.*F*std::pow(J, -2/3.)).reshaped();

    return deviatoric_gradientdF * deformationHessian;

}


template<int vertexDim, int elementDim>
Vector<double, elementDim*vertexDim> PseudostrainEnergy<vertexDim, elementDim>::compute_volume_gradient(
    const Matrix<double, vertexDim, vertexDim>& deformationGradient, const Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>& deformationHessian) const {
        
    const Matrix<double, vertexDim, vertexDim>& F = deformationGradient;
    Matrix<double, vertexDim, vertexDim> FinvT = F.inverse().transpose();

    double J = F.determinant();

    Matrix<double, 1, vertexDim * vertexDim> volume_gradientdF = 2/D_ * (J-1) * J * FinvT.reshaped();
    return volume_gradientdF * deformationHessian;
}


template<int vertexDim, int elementDim>
Matrix<double, elementDim*vertexDim, elementDim*vertexDim> PseudostrainEnergy<vertexDim, elementDim>::compute_hessian(
    const Matrix<double, elementDim, vertexDim>& vertices, 
    const std::vector<Matrix<double, vertexDim, vertexDim>>& deformationGradients,
    const std::vector<Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>>& deformationHessians,
    const MatrixXd& actuation) const {
    
    UNUSED(vertices); UNUSED(actuation);

    Matrix<double, elementDim*vertexDim, elementDim*vertexDim> hessian = Matrix<double, elementDim*vertexDim, elementDim*vertexDim>::Zero();

    // Normalized quadrature weights:
    int nquadpoints = deformationGradients.size();
    double quadratureWeights = 1. / nquadpoints; 
    
    // Compute integral over the element
    for (int s = 0; s < nquadpoints; ++s) {
        const auto& F = deformationGradients[s];
        const auto& nablaF = deformationHessians[s];
        hessian += (compute_deviatoric_hessian(F, nablaF) + compute_volume_hessian(F, nablaF)) * quadratureWeights;
    }  
    return hessian;  
}


template<int vertexDim, int elementDim>
Matrix<double, elementDim*vertexDim, elementDim*vertexDim> PseudostrainEnergy<vertexDim, elementDim>::compute_deviatoric_hessian(
    const Matrix<double, vertexDim, vertexDim>& deformationGradient, const Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>& deformationHessian) const {

        const Matrix<double, vertexDim, vertexDim>& F = deformationGradient;
        Matrix<double, vertexDim, vertexDim> FinvT = F.inverse().transpose();
        Matrix<double, 1, vertexDim*vertexDim> flattendFinvT = FinvT.reshaped();

        double J = F.determinant();
        double I_bar = (std::pow(J, -2/3.) * F.transpose() * F).trace();
    
        Matrix<double, 1, vertexDim * vertexDim> dI_bardF = \
            (-2/3.*std::pow(J, -2/3.)*((F.transpose() * F).trace())* FinvT + 2*F*std::pow(J, -2/3.)).reshaped();

        Matrix<double, vertexDim*vertexDim, vertexDim*vertexDim> part1 = dI_bardF.transpose() * dI_bardF * b_ * b_ * c_ * std::exp(b_*(I_bar-3));

        double traceFTF = (F.transpose() * F).trace();
        Matrix<double, 1, vertexDim*vertexDim> flattendF = F.reshaped();

        // d(J^(-2/3))/dF
        Matrix<double, 1, vertexDim*vertexDim> dJ23dF = -2/3. * std::pow(J, -2/3.) * FinvT.reshaped();

        // -2/3 *tr(FTF) *F^-T cross d(J^(-2/3))/dF
        Matrix<double, vertexDim*vertexDim, vertexDim*vertexDim> term1 = -2/3. * traceFTF * \
                flattendFinvT.transpose() * dJ23dF;

        // -2/3 * J^(-2/3) * F^-T cross d(tr(FTF))/dF
        Matrix<double, vertexDim*vertexDim, vertexDim*vertexDim> term2 = -4/3. * std::pow(J, -2/3.) * \
                flattendFinvT.transpose() * flattendF;

        // -2/3 * J^(-2/3) * tr(FTF) * M
        Matrix<double, vertexDim*vertexDim, vertexDim*vertexDim> FinvTGrad; 

        // Calculate -FinvT * E_i^T * FinvT for each E_i, then flatten the resulting matrix and put as a column in FinvTGrad
        for (unsigned i = 0; i < this->unitMatrices_.size(); ++i) {
                Matrix<double, vertexDim, vertexDim> blockMatrix = - FinvT * this->unitMatrices_[i].transpose() * FinvT;
                FinvTGrad.col(i) =  blockMatrix.reshaped();
        }
        Matrix<double, vertexDim*vertexDim, vertexDim*vertexDim> term3 = -2/3. * std::pow(J, -2/3.) * \
                traceFTF * FinvTGrad;

        // 2 * J^(-2/3) * Id
        Matrix<double, vertexDim*vertexDim, vertexDim*vertexDim> term4 = 2. * std::pow(J, -2/3.) * \
                Matrix<double, vertexDim*vertexDim, vertexDim*vertexDim>::Identity();

        // 2 * F cross d(J^(-2/3))/dF
        Matrix<double, vertexDim*vertexDim, vertexDim*vertexDim> term5 = 2 * \
                flattendF.transpose() * dJ23dF;

        Matrix<double, vertexDim*vertexDim, vertexDim*vertexDim> Hessian_d2F = (term1 + term2 + term3 + term4 + term5)*b_* c_* std::exp(b_*(I_bar-3)) + part1;        

        Matrix<double, elementDim*vertexDim, elementDim*vertexDim> deviatoricHessian = \
        (Hessian_d2F.transpose() * deformationHessian).transpose() * deformationHessian; 

        return deviatoricHessian;
}


template<int vertexDim, int elementDim>
Matrix<double, elementDim*vertexDim, elementDim*vertexDim> PseudostrainEnergy<vertexDim, elementDim>::compute_volume_hessian(
    const Matrix<double, vertexDim, vertexDim>& deformationGradient, const Matrix<double, vertexDim*vertexDim, vertexDim*elementDim>& deformationHessian) const {

        const Matrix<double, vertexDim, vertexDim>& F = deformationGradient;
        Matrix<double, vertexDim, vertexDim> FinvT = F.inverse().transpose();
        Matrix<double, 1, vertexDim * vertexDim> flattendFinvT = FinvT.reshaped();

        double J = F.determinant();

        Matrix<double, vertexDim*vertexDim, vertexDim*vertexDim> term1 = 2/D_*J*J*flattendFinvT.transpose()*flattendFinvT;
        Matrix<double, vertexDim*vertexDim, vertexDim*vertexDim> term2 = 2/D_*J*(J-1)*flattendFinvT.transpose()*flattendFinvT;

        Matrix<double, vertexDim*vertexDim, vertexDim*vertexDim> FinvTGrad; 
        // Calculate -FinvT * E_i^T * FinvT for each E_i, then flatten the resulting matrix and put as a column in FinvTGrad
        for (unsigned i = 0; i < this->unitMatrices_.size(); ++i) {
            Matrix<double, vertexDim, vertexDim> blockMatrix = - FinvT * this->unitMatrices_[i].transpose() * FinvT;
            FinvTGrad.col(i) =  blockMatrix.reshaped();
        }

        Matrix<double, vertexDim*vertexDim, vertexDim*vertexDim> term3 = 2/D_*J*(J-1)*FinvTGrad;
        Matrix<double, vertexDim*vertexDim, vertexDim*vertexDim> Hessian_d2F = term1 + term2 + term3;
        Matrix<double, elementDim*vertexDim, elementDim*vertexDim> volumeHessian = \
            (Hessian_d2F.transpose() * deformationHessian).transpose() * deformationHessian; 

        return volumeHessian;
}

template class PseudostrainEnergy <3, 4>;
template class PseudostrainEnergy <3, 8>;
