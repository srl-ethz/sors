#include "common.h"
#include "_test.h"

/**
 * @brief Entry point for running all unit tests.
 *
 * Executes the full test suite—including energy models, optimizers,
 * tetrahedral/hexahedral element routines, and constraint handling—
 * while timing the overall run and reporting final pass/fail status.
 */
int main () {
    std::cout << bcolors.HEADER << "Starting Unit Tests" << bcolors.ENDC << std::endl;

    // Unit Tests
    auto begin = std::chrono::steady_clock::now();
    
    TestEnergy testEnergy;
    testEnergy.test_all();

    TestOptimization testOptimization;
    testOptimization.test_all();
  
    TestTetrahedron testTetrahedron;
    testTetrahedron.test_all();

    TestCube testCube;
    testCube.test_all();

    TestConstraint testConstraint;
    testConstraint.test_all();

    // Print all stats
    testCube.print_statistics();

    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "Elapsed time: " << elapsed_ms.count() << " ms" << std::endl;

    if (testTetrahedron.all_passed()) {
        std::cout << bcolors.OKGREEN << "All tests passed!" << bcolors.ENDC << std::endl;
        return 0;
    } else {
        std::cout << bcolors.FAIL << "Some tests failed..." << bcolors.ENDC << std::endl;
        return 1;
    }
}