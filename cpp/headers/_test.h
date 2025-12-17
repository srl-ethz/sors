#ifndef _TEST_H
#define _TEST_H

#include "energy.h"
#include "solver.h"

/**
 * @class Testahedron
 * @brief Tetrahedral element used for dummy polynomial energy, gradient, and Hessian tests.
 *
 * Overrides the standard Tetrahedron energy with a hand-crafted x^2(x^2 - 0.5) potential
 * and corresponding analytical derivatives, enabling closed-form ground truth checks for
 * Energy and Solver unit tests.
 */
class Testahedron : public Tetrahedron {
public:
    Testahedron(Matrix<double, TET_E_DIM, TET_V_DIM> &vertices);

    // Computes energy based on the element's vertices
    virtual double compute_energy(
        Matrix<double, TET_E_DIM, TET_V_DIM>& vertices,
        Params& actuation) const override;

    // Computes gradient of the energy function with respect to vertices
    Vector<double, TET_EV_DIM> compute_gradient(
        Matrix<double, TET_E_DIM, TET_V_DIM>& vertices,
        Params& actuation) const override;

    // Computes Hessian matrix of the energy function with respect to vertices
    Matrix<double, TET_EV_DIM, TET_EV_DIM> compute_hessian(
        Matrix<double, TET_E_DIM, TET_V_DIM>& vertices,
        Params& actuation) const override;
};

/**
 * @class TestClass
 * @brief Base class for all unit test suites.
 *
 * Provides common bookkeeping for test results (count, pass/fail), a uniform test_all()
 * entry point, and simple reporting utilities for aggregated statistics.
 */
class TestClass {
public:
    static const int MAX_TESTS = 14;
    TestClass () {};

    virtual bool test_all() = 0;
    void add_result(bool result, std::string testName);
    void print_statistics(){
        std::cout << bcolors.OKGREEN << "Tests Passed: " << testPassed << "/" << testCount << bcolors.ENDC << std::endl;
    }
    bool all_passed(){
        return testPassed == testCount;
    }

    MatrixXd vertices_;
    MatrixXi eleIdx_;

protected:
    // Static variable to keep track of the number of tests
    static int testCount;
    static int testPassed;
    static int testFailed;
};

/**
 * @class TestEnergy
 * @brief Unit tests for the global Energy class on a small tetrahedral mesh.
 *
 * Builds a toy mesh, replaces physical elements by Testahedron instances, and verifies
 * correctness of global energy, gradient, and Hessian assembly as well as basic
 * initialization of the Energy system.
 */
class TestEnergy : public TestClass{
public:
    static const int VDIM = 3;
    static const int EDIM = 4;
    static const int NUM_VERTICES = 5;
    static const int NUM_ELEMENTS = 2;
    static constexpr double TOL = 1e-9;

    TestEnergy();

    bool test_all(){
        add_result(test_init(), "Local element vertices");
        add_result(test_energy(), "Dummy global energy computation");
        add_result(test_gradient(), "Dummy global gradient computation");
        add_result(test_hessian(), "Dummy global Hessian computation");
        return true;
    }

protected:
    Matrix<double, -1, VDIM> vertices_;
    Matrix<int, -1, EDIM> eleIdx_;
    Energy<VDIM, EDIM> systemEnergy_;

    bool test_init();     // Not const since we construct a system_energy object, which requires a pass-by-reference.
    bool test_energy();   // Not const since we reshape
    bool test_gradient(); // Not const since we reshape
    bool test_hessian();  // Not const since we reshape
};


/**
 * @class TestOptimization
 * @brief Unit tests for the non-linear solver / optimization routines.
 *
 * Reuses the TestEnergy setup and checks that the Newton-based optimizer converges
 * to a (near) optimal configuration by validating that the final gradient norm is
 * close to zero.
 */
class TestOptimization : private TestEnergy {
public:
    TestOptimization() : TestEnergy() {};
    bool test_all() {
        add_result(test_optimization(), "Optimization algorithm");
        return true;
    }

private:
    bool test_optimization(); // Not const because of pass-by-reference
};

/**
 * @class TestTetrahedron
 * @brief Unit tests for a single Tetrahedron element.
 *
 * Verifies closed-form quantities such as the deformation Hessian dF/dx, the
 * gravitational energy gradient, and that energy and gradient vanish for an
 * undeformed, gravity-free configuration, as well as gravitational energy evaluation.
 */
class TestTetrahedron : public TestClass {
public:
    static const int VDIM = 3;
    static const int EDIM = 4;
    static constexpr double TOL = 1e-9;

    TestTetrahedron();

    bool test_all() {
        add_result(test_deformationHessian(), "Deformation hessian");
        add_result(test_energy_zero(), "Energy computation all zero");
        add_result(test_gradient_zero(), "Gradient computation all zero");
        return true;
    }

protected:
    Matrix<double, EDIM, VDIM> undeformedVertices_;
    Matrix<double, EDIM, VDIM> currentVertices_;
    double dt_;

    bool test_deformationHessian();
    bool test_gravitationalGradient();
    bool test_energy_zero();
    bool test_energy_grav();
    bool test_gradient_zero();
};

/**
 * @class TestCube
 * @brief Unit tests on a cube discretized into tetrahedra for multi-element behavior.
 *
 * Exercises a variety of physical effects on a small block, including energy
 * minimization under load, gravity-driven motion, pressure forces, arbitrary
 * vertex forces, and SoftCon-style muscle actuation, by checking center-of-mass
 * motion and deformation trends.
 */
class TestCube : public TestClass {
public:
    static const int VDIM = 3;
    static const int EDIM = 4;
    static constexpr double TOL = 1e-3;

    TestCube();

    bool test_all () {
        add_result(test_energy_min(), "General energy minimization check");
        add_result(test_gravity_force(), "Gravity force validation");
        add_result(test_pressure_force(), "Pressure force validation");
        add_result(test_external_force(), "External force validation");
        add_result(test_softcon_muscle(), "Softcon muscle validation");
        return true;
    }

protected:
    Matrix<double, -1, VDIM> vertices_;
    Matrix<int, -1, EDIM> elements_;
    double dt_;

    bool test_energy_min();
    bool test_gravity_force();
    bool test_pressure_force();
    bool test_external_force();
    bool test_softcon_muscle();
};


/**
 * @class TestConstraint
 * @brief Unit tests for constraint handling, in particular plane contact.
 *
 * Sets up a single tetrahedron partially penetrating a contact plane and verifies
 * that a single constrained SQP step resolves penetration so that all surface
 * vertices satisfy the non-penetration condition.
 */
class TestConstraint : public TestClass {
public:
    static const int VDIM = 3;
    static const int EDIM = 4;
    static constexpr double TOL = 1e-4;

    TestConstraint() {}

    bool test_all() {
        add_result(test_plane_contact_no_penetration(), "Plane contact: no penetration after one step");
        return true;
    }

protected: 
    bool test_plane_contact_no_penetration();
};


#endif
