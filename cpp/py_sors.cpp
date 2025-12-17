#include "common.h"
#include "params.h"
#include "io.h"
#include "energy.h"
#include "solver.h"
#include "demos.h"
#include "plane3D.h" 
#include "disk3D.h"        
#include "diskContact.h"   

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

/**
 * @module py_sors
 * @brief Python bindings for the SORS simulation framework.
 *
 * Exposes high-performance C++ components—energy models, nonlinear solvers,
 * time integrators, collision constraints, and VTU/PVD I/O—to Python.  
 * Designed for rapid prototyping of soft-robotics simulations while retaining
 * the speed and numerical robustness of the underlying C++ engine.
 */
PYBIND11_MODULE (py_sors, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("run_msd_timestepping", &run_msd_timestepping, "Run msd demo");

    // Define Energy
    m.def("Energy", []( 
        Matrix<double, -1, 3>& undeformedVertices,
        py::object& eleIdx,
        SimulationSettings settings,
        Vector<double, 3> gravAcceleration,
        double dampingAlpha,
        std::vector<std::set<std::string>> elementEnergiesList,
        std::vector<Params> elementParameterList,
        std::vector<std::string> constraintTypesList,
        Params constraintParameterList,
        std::vector<std::string> forceTypesList,
        Params forceParameterList
    ) -> py::object {
        if (settings.meshType == "Tetrahedron") {
            auto eleIdxMat = eleIdx.cast<Matrix<int, -1, 4>>();
            auto ptr = std::unique_ptr<Energy<3,4>>(new Energy<3,4>(undeformedVertices, eleIdxMat, settings,
                gravAcceleration, dampingAlpha, elementEnergiesList, elementParameterList,
                constraintTypesList, constraintParameterList,
                forceTypesList, forceParameterList));
            return py::cast(std::move(ptr));
        }
        else if (settings.meshType == "Hexahedron") {
            auto eleIdxMat = eleIdx.cast<Matrix<int, -1, 8>>();
            auto ptr = std::unique_ptr<Energy<3,8>>(new Energy<3,8>(undeformedVertices, eleIdxMat, settings,
                gravAcceleration, dampingAlpha, elementEnergiesList, elementParameterList,
                constraintTypesList, constraintParameterList,
                forceTypesList, forceParameterList));
            return py::cast(std::move(ptr));
        }
        else {
            throw std::runtime_error("Unknown mesh type: " + settings.meshType);
        }
    });
    // The following instances are needed for pybind to figure out which C++ classes to associate to the previously returned unknown py::object types.
    py::class_<Energy<3,4>>(m, "EnergyTet")
        .def(py::init< 
            Matrix<double, -1, 3>&,
            Matrix<int, -1, 4>&,
            SimulationSettings,
            Vector<double, 3>,
            double,
            std::vector<std::set<std::string>>,
            std::vector<Params>,
            std::vector<std::string>,
            Params,
            std::vector<std::string>,
            Params
        >(), py::arg("undeformedVertices"), py::arg("eleIdx"), py::arg("settings"), py::arg("gravAcceleration"), py::arg("dampingAlpha")=0.0, py::arg("elementEnergiesList"), py::arg("elementParameterList"), py::arg("constraintTypesList"), py::arg("constraintParameterList"), py::arg("forceTypesList"), py::arg("forceParameterList"))
        .def("set_initial_velocity", &Energy<3, 4>::set_initial_velocity)
        .def("set_initial_deformation", &Energy<3, 4>::set_initial_deformation);
    py::class_<Energy<3,8>>(m, "EnergyHex")
        .def(py::init< 
            Matrix<double, -1, 3>&,
            Matrix<int, -1, 8>&,
            SimulationSettings,
            Vector<double, 3>,
            double,
            std::vector<std::set<std::string>>,
            std::vector<Params>,
            std::vector<std::string>,
            Params,
            std::vector<std::string>,
            Params
        >(), py::arg("undeformedVertices"), py::arg("eleIdx"), py::arg("settings"), py::arg("gravAcceleration"), py::arg("dampingAlpha")=0.0, py::arg("elementEnergiesList"), py::arg("elementParameterList"), py::arg("constraintTypesList"), py::arg("constraintParameterList"), py::arg("forceTypesList"), py::arg("forceParameterList"))
        .def("set_initial_velocity", &Energy<3, 8>::set_initial_velocity)
        .def("set_initial_deformation", &Energy<3, 8>::set_initial_deformation);

        
    // Define Solver
    m.def("Solver", [](SimulationSettings settings) -> py::object {
        if (settings.meshType == "Tetrahedron") {
            auto ptr = std::unique_ptr<Solver<3,4>>(new Solver<3,4>());
            return py::cast(std::move(ptr));
        }
        else if (settings.meshType == "Hexahedron") {
            auto ptr = std::unique_ptr<Solver<3,8>>(new Solver<3,8>());
            return py::cast(std::move(ptr));
        }
        else {
            throw std::runtime_error("Unknown mesh type: " + settings.meshType);
        }
    });

    
    // The following instances are needed for pybind to figure out which C++ classes to associate to the previously returned unknown py::object types.
    py::class_<Solver<3, 4>>(m, "SolverTet")
        .def("step", static_cast<VectorXd (Solver<3, 4>::*)(
            VectorXd&, Energy<3, 4>&, Params&, double, int) const>(&Solver<3, 4>::step),
            "Step solution forward in time.",
            py::arg("solution"),
            py::arg("systemEnergy"),
            py::arg("act"),
            py::arg("dt")=0.0,
            py::arg("substeps")=0)
        .def("step", static_cast<VectorXd (Solver<3, 4>::*)(
            VectorXd&, Energy<3, 4>&, double, int) const>(&Solver<3, 4>::step),
            "Step solution forward in time with default actuation.",
            py::arg("solution"),
            py::arg("systemEnergy"),
            py::arg("dt")=0.0,
            py::arg("substeps")=0);


    py::class_<Solver<3, 8>>(m, "SolverHex")
        .def("step", static_cast<VectorXd (Solver<3, 8>::*)(
            VectorXd&, Energy<3, 8>&, Params&, double, int) const>(&Solver<3, 8>::step),
            "Step solution forward in time.",
            py::arg("solution"),
            py::arg("systemEnergy"),
            py::arg("act"),
            py::arg("dt")=0.0,
            py::arg("substeps")=0)
        .def("step", static_cast<VectorXd (Solver<3, 8>::*)(
            VectorXd&, Energy<3, 8>&, double, int) const>(&Solver<3, 8>::step),
            "Step solution forward in time with default actuation.",
            py::arg("solution"),
            py::arg("systemEnergy"),
            py::arg("dt")=0.0,
            py::arg("substeps")=0);


    py::class_<Params>(m, "Params")
        .def(py::init<>())
        .def(py::init<std::vector<std::string>, std::vector<MatrixXd>>())
        .def(py::init<std::vector<std::string>, std::vector<double>>())
        .def("get_value", &Params::get_value)
        .def("set_param", static_cast<void (Params::*)(std::string, MatrixXd)>(&Params::set_param))
        .def("set_param", static_cast<void (Params::*)(std::string, std::vector<double>)>(&Params::set_param))
        .def("set_param", static_cast<void (Params::*)(std::string, double)>(&Params::set_param))
        .def("set_param", static_cast<void (Params::*)(const std::string&)>(&Params::set_param));


    py::class_<SimulationSettings>(m, "SimulationSettings")
        .def(py::init<>())
        .def_readwrite("description", &SimulationSettings::description)
        .def_readwrite("verbose", &SimulationSettings::verbose)
        .def_readwrite("meshFilePath", &SimulationSettings::meshFilePath)
        .def_readwrite("outputFolder", &SimulationSettings::outputFolder)
        .def_readwrite("meshFileName", &SimulationSettings::meshFileName)
        .def_readwrite("meshType", &SimulationSettings::meshType)
        .def_readwrite("dt", &SimulationSettings::dt)
        .def_readwrite("numSteps", &SimulationSettings::numSteps)
        .def_readwrite("substeps", &SimulationSettings::substeps)
        .def_readwrite("CFLtimeSteppingFlag", &SimulationSettings::CFLtimeSteppingFlag)
        .def_readwrite("timeSteppingScheme", &SimulationSettings::timeSteppingScheme)
        .def_readwrite("solverMethod", &SimulationSettings::solverMethod)
        .def_readwrite("numThreads", &SimulationSettings::numThreads)
        .def("verbose_print", &SimulationSettings::verbose_print);

        
    py::class_<IO>(m, "IO")
        .def_static("save_tet_VTU", &IO::save_tet_VTU, 
            "Save tetrahedral mesh (vertices and elements) to VTU file.", 
            py::arg("filename"), py::arg("vertices"), py::arg("elements"), py::arg("visuals") = std::vector<VisualOption>(), py::arg("planes")  = std::vector<Plane3D>())
        .def_static("save_hex_VTU", &IO::save_hex_VTU, 
            "Save hexahedral mesh (vertices and elements) to VTU file.", 
            py::arg("filename"), py::arg("vertices"), py::arg("elements"), py::arg("visuals") = std::vector<VisualOption>(), py::arg("planes")  = std::vector<Plane3D>())
        .def_static("save_pvd", &IO::save_pvd, 
            "Save PVD file summarizing all VTUs.", 
            py::arg("filename"), py::arg("outputFolder"), py::arg("timesteps")=1, py::arg("dt")=1)
        .def_static("extract_tet_surfaces", &IO::extract_tet_surfaces, 
            "Extract surfaces from tetrahedral mesh.", 
            py::arg("vertices"), py::arg("eleIdx"), py::arg("verboseFlag")=false)
        .def_static("extract_hex_surfaces", &IO::extract_hex_surfaces, 
            "Extract surfaces from hexahedral mesh.", 
            py::arg("vertices"), py::arg("eleIdx"), py::arg("verboseFlag")=false)
        .def_static("hex_to_5_tets", &IO::hex_to_5_tets, 
            "Converts hexahedral elements to 5 tetrahedra.",
            py::arg("hexElements"));


    py::enum_<VisualType>(m, "VisualType")
        .value("VectorVertex",  VisualType::VectorVertex)
        .value("VectorElement", VisualType::VectorElement)
        .value("ScalarVertex",  VisualType::ScalarVertex)
        .value("ScalarElement", VisualType::ScalarElement);


    py::class_<VisualOption>(m, "VisualOption")
        .def(py::init<>())
        .def_readwrite("name",      &VisualOption::name)
        .def_readwrite("type",      &VisualOption::type)
        .def_readwrite("data",      &VisualOption::data)      
        .def_readwrite("threshold", &VisualOption::threshold);


    py::class_<Plane3D>(m, "Plane3D")
        .def(py::init<>())
        .def(py::init<Vector3d, Vector3d>(), py::arg("point"), py::arg("normal"))
        .def("is_point_on_plane", &Plane3D::is_point_on_plane)
        .def("distance_to_point", &Plane3D::distance_to_point)
        .def("project_point_on_plane", &Plane3D::project_point_on_plane)
        .def("create_square_points", &Plane3D::create_square_points, py::arg("squareSide"))
        .def("get_point", &Plane3D::get_point)
        .def("set_point", &Plane3D::set_point)
        .def("get_normal", &Plane3D::get_normal)
        .def("set_normal", &Plane3D::set_normal)
        .def("get_square_points", &Plane3D::get_square_points);


    // Exposes DiskContact class and its set_disk method to Python for PokeFlex demo
    py::class_<DiskContact<3,4>>(m, "DiskContact")
    .def("set_disk", [](DiskContact<3,4>& self,
        const Eigen::Vector3d& center,
        const Eigen::Vector3d& normal,
        double radius, int diskIdx)
        {
            Eigen::VectorXd c(3), n(3);
            c = center;
            n = normal; // normalize in Python if you want
            self.set_disk(c, n, radius, diskIdx);
        },
        py::arg("center"),
        py::arg("normal"),
        py::arg("radius"),
        py::arg("diskIdx") = 0);

        
    // Function to get DiskContact constraint from Energy object by index
    m.def("get_disk_contact_constraint",
      [](Energy<3,4>& energy, int idx) -> DiskContact<3,4>* {
          // Assumes energy.constraints_ is public
          if (idx < 0 || idx >= static_cast<int>(energy.constraints_.size()))
              throw py::index_error("constraint index out of range");
          auto* base = energy.constraints_[idx].get();
          auto* dc = dynamic_cast<DiskContact<3,4>*>(base);
          if (!dc) throw py::value_error("constraint at index is not DiskContact");
          return dc; // Non-owning; lifetime tied to Energy
      },
      py::arg("energy"),
      py::arg("idx"),
      py::return_value_policy::reference);

}