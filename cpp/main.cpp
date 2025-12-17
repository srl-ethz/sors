#include "demos.h"

int main() {
    // Start Multithreaded Simulation
    omp_set_dynamic(0);
    omp_set_num_threads(10);

    // Run simulation
    auto begin = std::chrono::steady_clock::now();

    // DEMOS

    // Soft bodies and soft robots
    // run_monkey("cpp/demos/meshes/monkey.msh", "output", 400);
    // run_helicoid_arm("cpp/demos/meshes/helicoid_2segments.msh", "output", 40);
    // run_msd_timestepping("cpp/demos/meshes/msd_cylinder_v3.msh", "output", 400);
    // run_sopra_arm("cpp/demos/meshes/arm.msh", "output", 400);

    // Constraint-based simulations
    run_impact_duck("cpp/demos/meshes/duck.msh", "output", 800);
    // run_bouncing_sphere("cpp/demos/meshes/sphere1.msh", "output", 2700);
    // run_squeeze_sphere("cpp/demos/meshes/sphere1.msh", "output", 450);
    // run_multiple_constraints_sphere("cpp/demos/meshes/sphere1.msh", "output", 500);

    // SoftCon Muscle
    // run_softcon_muscle("cpp/demos/meshes/softcon_cylinder.msh", "output", 100);

    // Beam under gravity, hex or tet elements
    // run_beam_under_gravity<8>("beam_hex", "output", 150);
    // run_beam_under_gravity<4>("beam_tet", "output", 150);


    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "Elapsed time: " << elapsed_ms.count() / 1000. << " s" << std::endl;

    return 0;
}