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

add_executable(emberfield_tests version_test.cpp DataStructure/Storage/data_structure_test.cpp Numerics/LinearAlgebra/linear_algebra_test.cpp Solver/Iterative/solver_test.cpp Solver/Iterative/cg_test.cpp IO/Matrix/matrix_market_test.cpp DataStructure/Mesh/mesh_test.cpp IO/Mesh/msh_test.cpp Simulation/Case/case_test.cpp IO/Case/case_io_test.cpp)
target_link_libraries(emberfield_tests PRIVATE EmberField::emberfield GTest::gtest_main)
target_sources(emberfield_tests PRIVATE nvtx_profiler_test.cpp)
target_sources(emberfield_tests PRIVATE Simulation/Case/temporal_test.cpp)
target_sources(emberfield_tests PRIVATE IO/Result/result_test.cpp)
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    if(Python3_VERSION VERSION_GREATER_EQUAL "3.10")
        add_test(NAME case-conversion-unit COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/IO/Case/test_conversion.py")
        add_test(NAME case-conversion-integration COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/IO/Case/test_conversion.py" --executable "$<TARGET_FILE:emberfield_cli>")
    endif()
    if(EMBERFIELD_BUILD_TUI AND UNIX)
        add_test(NAME cli-tui-contract COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/IO/Case/test_tui.py" "$<TARGET_FILE:emberfield_cli>"
            "${PROJECT_SOURCE_DIR}/example/cases")
        set_tests_properties(cli-tui-contract PROPERTIES TIMEOUT 120)
    endif()
    add_test(NAME cli-result-contract COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/IO/Result/validate_cli.py"
        "$<TARGET_FILE:emberfield_cli>" "${PROJECT_SOURCE_DIR}/example/cases")
endif()
target_include_directories(emberfield_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_sources(emberfield_tests PRIVATE FEM/Element/reference_element_test.cpp FEM/Space/finite_element_space_test.cpp Simulation/Case/temperature_space_test.cpp FEM/Element/geometry_integration_test.cpp Simulation/Thermal/thermal_element_test.cpp Simulation/Case/condition_preparation_test.cpp Simulation/Thermal/gpu_thermal_test.cpp Simulation/Case/gpu_preparation_test.cpp)
target_sources(emberfield_tests PRIVATE IO/Case/tui_progress_test.cpp)
target_compile_definitions(emberfield_tests PRIVATE EMBERFIELD_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
include(GoogleTest)
target_sources(emberfield_tests PRIVATE FEM/Assembly/assembly_test.cpp Numerics/LinearSystem/system_test.cpp Simulation/Case/thermal_assembly_test.cpp)
gtest_discover_tests(emberfield_tests NO_PRETTY_VALUES)

add_executable(emberfield_amg_tests Solver/Preconditioner/amg_internal_test.cu
    Solver/Preconditioner/amg_move_test.cpp
    Solver/Preconditioner/NativeAggregationOracle.cu)
target_link_libraries(emberfield_amg_tests PRIVATE EmberField::emberfield GTest::gtest_main)
foreach(cudaInclude IN LISTS CUDAToolkit_INCLUDE_DIRS)
    if(EXISTS "${cudaInclude}/cccl/cub/cub.cuh")
        target_include_directories(emberfield_amg_tests PRIVATE "${cudaInclude}/cccl")
    endif()
endforeach()
gtest_discover_tests(emberfield_amg_tests NO_PRETTY_VALUES)

target_sources(emberfield_tests PRIVATE Numerics/LinearAlgebra/optimization_test.cpp)
