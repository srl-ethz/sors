#include "integrator.h"

double TimeIntegrator::integrate_energy (const VectorXd& q, const VectorXd& mass, const VectorXd& gravAcceleration, double dampingAlpha, double dt, double energyInt) const {
    // Compute the total energy by adding the kinetic energy and gravity contribution based on the time stepping scheme. The kinetic, damping, and gravity together form the inertial energy.

    // Mass times acceleration times position, where mass is defined in each dimension for each vertex.
    double gravitationalEnergy = -(
        (mass.cwiseProduct(q)).reshaped<RowMajor>(q.size()/3, 3).array().rowwise() * gravAcceleration.transpose().array()
    ).sum();

    // By default apply gravity as well.
    if (dt == 0)
        return gravitationalEnergy + energyInt;
    
    // Kinetic and Damping Energy
    if (this->timeSteppingScheme_ == "backward_euler") {
        VectorXd v = (q - qPrev_) / dt;
        VectorXd a = (v - vPrev_) / dt;

        double kineticEnergy = 0.5 * mass.cwiseProduct(a).dot(a) * dt * dt;
        double dampingEnergy = 0.5 * dampingAlpha * dt * mass.cwiseProduct(v).dot(v);

        double inertialEnergy = kineticEnergy + gravitationalEnergy + dampingEnergy;
        double totalEnergy = inertialEnergy + energyInt;
        return totalEnergy;
    }
    else if (this->timeSteppingScheme_ == "crank_nicolson") {
        VectorXd v = 2 * (q - qPrev_) / dt - vPrev_;
        VectorXd a = (v - vPrev_) / dt;

        double kineticEnergy = 0.5 * mass.cwiseProduct(a).dot(a) * dt * dt / 2;
        double dampingEnergy = 0.25 * dampingAlpha * dt * mass.cwiseProduct(v).dot(v);

        double inertialEnergy = kineticEnergy + gravitationalEnergy + dampingEnergy;
        double totalEnergy = inertialEnergy + energyInt;
        return totalEnergy;
    }
    else {
        std::cerr << bcolors.FAIL << "Unknown time stepping scheme: " << this->timeSteppingScheme_ << bcolors.ENDC << std::endl;
        assert(false);
    }
}


VectorXd TimeIntegrator::integrate_gradient (const VectorXd& q, const VectorXd& mass, const VectorXd& gravAcceleration, double dampingAlpha, double dt, const VectorXd& gradientInt) {
    // Compute the total gradient by adding the inertial energy contribution based on the time stepping scheme.

    // Element-wise product of mass and gravity acceleration
    VectorXd gravitationalGradient = -(
        mass.reshaped<RowMajor>(q.size()/3, 3).array().rowwise() * gravAcceleration.transpose().array()
    ).reshaped<RowMajor>();

    // By default apply gravity as well.
    if (dt == 0)
        return gravitationalGradient + gradientInt;

    this->gPrevTmp_ = gradientInt; // Store the internal force at the current step temporarily

    // Kinetic gradient
    if (this->timeSteppingScheme_ == "backward_euler") {
        VectorXd v = (q - qPrev_) / dt;

        VectorXd kineticGradient = mass.cwiseProduct((v - vPrev_) / dt);
        VectorXd dampingGradient = dampingAlpha * mass.cwiseProduct(v);

        VectorXd inertialGradient = kineticGradient + gravitationalGradient + dampingGradient;
        VectorXd totalGradient = inertialGradient + gradientInt;
        return totalGradient;
    }
    else if (this->timeSteppingScheme_ == "crank_nicolson") {
        VectorXd v = 2 * (q - qPrev_) / dt - vPrev_;

        VectorXd kineticGradient = mass.cwiseProduct((v - vPrev_) / dt);
        VectorXd dampingGradient = dampingAlpha * mass.cwiseProduct(v);

        VectorXd inertialGradient = kineticGradient + gravitationalGradient + dampingGradient;
        VectorXd totalGradient = inertialGradient + 0.5 * (gradientInt + gPrev_);
        return totalGradient;
    }
    else {
        std::cerr << bcolors.FAIL << "Unknown time stepping scheme: " << this->timeSteppingScheme_ << bcolors.ENDC << std::endl;
        assert(false);
    }
}


SparseMatrix<double> TimeIntegrator::integrate_hessian (const VectorXd& q, const VectorXd& mass, const VectorXd& gravAcceleration, double dampingAlpha, double dt, const SparseMatrix<double>& hessianInt) const {
    // Compute the total hessian by adding the kinetic energy contribution based on the time stepping scheme.
    UNUSED(gravAcceleration);
    if (dt == 0)
        return hessianInt;

    // Kinetic and Damping Hessian
    if (this->timeSteppingScheme_ == "backward_euler") {
        SparseMatrix<double> inertialHessian(q.size(), q.size());
        std::vector<Triplet<double>> tripletList(q.size());
        #pragma omp parallel for
        for (int i = 0; i < q.size(); i++) {
            tripletList[i] = Triplet<double>(
                i, i, 
                mass(i) / (dt * dt) + dampingAlpha * mass(i) / dt
            );
        }
        inertialHessian.setFromTriplets(tripletList.begin(), tripletList.end());
        SparseMatrix<double> totalHessian = inertialHessian + hessianInt;
        return totalHessian;
    }
    else if (this->timeSteppingScheme_ == "crank_nicolson") {
        SparseMatrix<double> kineticHessian(q.size(), q.size());
        std::vector<Triplet<double>> tripletList(q.size());
        #pragma omp parallel for
        for (int i = 0; i < q.size(); i++) {
            tripletList[i] = Triplet<double>(
                i, i, 
                2 * mass(i) / (dt * dt) + 2 * dampingAlpha * mass(i) / dt
            );
        }
        kineticHessian.setFromTriplets(tripletList.begin(), tripletList.end());
        SparseMatrix<double> totalHessian = kineticHessian + 0.5 * hessianInt;
        return totalHessian;
    }
    else {
        std::cerr << bcolors.FAIL << "Unknown time stepping scheme: " << this->timeSteppingScheme_ << bcolors.ENDC << std::endl;
        assert(false);
    }
}