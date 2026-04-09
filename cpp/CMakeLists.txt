cmake_minimum_required(VERSION 3.15)

project(sors)

set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")

# macOS OpenMP support via Homebrew libomp
if(APPLE)
    execute_process(COMMAND brew --prefix libomp OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(HOMEBREW_LIBOMP_PREFIX)
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()

# Include external libraries
include_directories(SYSTEM "../external/eigen")
include_directories("../external/vtu11")

# Add headers
include_directories("./headers")
include_directories("./headers/energies")
include_directories("./headers/constraints")
include_directories("./headers/forces")
include_directories("./headers/geometries")
include_directories("./demos")

# External libraries
# PBRT is optional (only needed for rendering), skip on macOS due to compatibility issues
# include(ExternalProject)
# ExternalProject_Add(pbrt ...)

# MshIO
add_subdirectory(../external/MshIO MshIO)
target_compile_options(mshio PRIVATE -w) # Cannot do anything about these warnings anyways, so we disable.
find_package(OpenMP REQUIRED)

# OSQP
add_subdirectory(../external/osqp osqp)

# Collect all source files
file(GLOB_RECURSE CORE_CPP
    "./src/*.cpp"
    "./demos/*.cpp"
    "./src/energies/*.cpp"
    "./src/constraints/*.cpp"
    "./src/forces/*.cpp"
    "./src/geometries/*.cpp"
)

# Create the static library
add_library(sim_core STATIC ${CORE_CPP})
target_link_libraries(sim_core OpenMP::OpenMP_CXX mshio::mshio osqpstatic)

# Main executable
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} sim_core)

# Compile unit tests
add_executable(unit_test unit_test.cpp)
target_link_libraries(unit_test sim_core)

# Find current terminal Python3.
execute_process(COMMAND bash -c "python3-config --includes|sed 's/-I//g'|awk '{print $1;}'" OUTPUT_VARIABLE PYTHON_INCLUDES)
include_directories(${PYTHON_INCLUDES})
# Pybind11
add_subdirectory(../external/pybind11 pybind11)
pybind11_add_module(py_${PROJECT_NAME} py_sors.cpp)

# Pybind requires position-independent code.
set_target_properties(mshio PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(sim_core PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(osqp PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
target_link_libraries(py_${PROJECT_NAME} PRIVATE sim_core)

install(TARGETS py_${PROJECT_NAME} LIBRARY DESTINATION . COMPONENT python_module)

# Make sure the module can be imported as "import py_sors"
set(INIT_PY_CONTENT "from .py_${PROJECT_NAME} import *")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/__init__.py" "${INIT_PY_CONTENT}")
