find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    set(INSTALL_GTEST OFF CACHE BOOL "Install GoogleTest" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "Use shared CRT" FORCE)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(thermofem_tests version_test.cpp DataStructure/Storage/data_structure_test.cpp Numerics/LinearAlgebra/linear_algebra_test.cpp Solver/Iterative/solver_test.cpp Solver/Iterative/cg_test.cpp IO/Matrix/matrix_market_test.cpp DataStructure/Mesh/mesh_test.cpp IO/Mesh/msh_test.cpp Simulation/Case/case_test.cpp IO/Case/case_io_test.cpp)
target_link_libraries(thermofem_tests PRIVATE ThermoFEM::thermofem GTest::gtest_main)
target_include_directories(thermofem_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_sources(thermofem_tests PRIVATE FEM/Element/reference_element_test.cpp FEM/Space/finite_element_space_test.cpp Simulation/Case/temperature_space_test.cpp FEM/Element/geometry_integration_test.cpp Simulation/Thermal/thermal_element_test.cpp Simulation/Case/condition_preparation_test.cpp Simulation/Thermal/gpu_thermal_test.cpp Simulation/Case/gpu_preparation_test.cpp)
include(GoogleTest)
target_sources(thermofem_tests PRIVATE FEM/Assembly/assembly_test.cpp Numerics/LinearSystem/system_test.cpp Simulation/Case/thermal_assembly_test.cpp)
gtest_discover_tests(thermofem_tests NO_PRETTY_VALUES)

add_executable(thermofem_amg_tests Solver/Preconditioner/amg_internal_test.cu
    Solver/Iterative/solver_test.cpp
    Solver/Preconditioner/NativeAggregationOracle.cu)
target_link_libraries(thermofem_amg_tests PRIVATE ThermoFEM::thermofem GTest::gtest_main)
foreach(cudaInclude IN LISTS CUDAToolkit_INCLUDE_DIRS)
    if(EXISTS "${cudaInclude}/cccl/cub/cub.cuh")
        target_include_directories(thermofem_amg_tests PRIVATE "${cudaInclude}/cccl")
    endif()
endforeach()
gtest_discover_tests(thermofem_amg_tests NO_PRETTY_VALUES)
