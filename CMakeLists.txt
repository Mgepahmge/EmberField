cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures to build")
endif()

project(EmberField VERSION 0.1.0 DESCRIPTION "CUDA finite element heat conduction solver" LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)
find_package(CUDAToolkit 12.0 REQUIRED)
option(EMBERFIELD_BUILD_EXAMPLES "Build EmberField examples" ON)

add_library(emberfield STATIC src/version.cu src/DataStructure/Storage/Memory.cu src/DataStructure/Storage/CsrMatrix.cu src/Numerics/LinearAlgebra/Backend.cu src/Numerics/LinearAlgebra/IterationContext.cu src/IO/Matrix/MatrixMarket.cu src/Solver/Preconditioner/Amg.cu src/DataStructure/Mesh/Mesh.cu src/IO/Mesh/Msh.cu
    src/Simulation/Case/Diagnostics.cu src/Simulation/Case/Validation.cu src/Simulation/Case/Region.cu src/Simulation/Case/Binding.cu src/DataStructure/Mesh/FacetTopology.cu src/IO/Case/Case.cu
    src/FEM/Element/ReferenceElement.cu src/FEM/Space/Space.cu src/Simulation/Case/TemperatureSpace.cu
    src/FEM/Element/EvaluationError.cu src/FEM/Element/Geometry.cu src/FEM/Element/Quadrature.cu src/FEM/Element/Integration.cu
    src/Simulation/Thermal/ElementOperations.cu src/Simulation/Case/Conditions.cu src/Simulation/Case/Preparation.cu src/Simulation/Thermal/GpuElementOperations.cu src/Simulation/Case/GpuPreparation.cu)
add_library(EmberField::emberfield ALIAS emberfield)
target_sources(emberfield PRIVATE src/DataStructure/Field/Snapshot.cu src/IO/Result/Vtu.cu src/IO/Result/Pvd.cu)
target_sources(emberfield PRIVATE src/IO/Case/Write.cu src/Simulation/Thermal/PostprocessOperations.cu)
target_sources(emberfield PRIVATE src/Simulation/Case/Postprocess.cu)
target_sources(emberfield PRIVATE src/Simulation/Case/OutputOptions.cu)
target_sources(emberfield PRIVATE src/IO/Case/CommandLine.cu src/IO/Result/Archive.cu src/Simulation/Case/Run.cu)
target_sources(emberfield PRIVATE src/FEM/Assembly/Plan.cu src/FEM/Assembly/Operations.cu src/Numerics/LinearSystem/Operations.cu src/Simulation/Case/Assembly.cu src/Simulation/Case/System.cu)
target_include_directories(emberfield SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third-party/picojson")
target_include_directories(emberfield PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(emberfield PUBLIC cxx_std_20 cuda_std_20)
target_link_libraries(emberfield PUBLIC CUDA::cudart CUDA::cublas CUDA::cusparse)
foreach(cudaInclude IN LISTS CUDAToolkit_INCLUDE_DIRS)
    if(EXISTS "${cudaInclude}/cccl/cub/cub.cuh")
        target_include_directories(emberfield PRIVATE "${cudaInclude}/cccl")
    endif()
endforeach()
set_target_properties(emberfield PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(emberfield_cli src/main.cu)
target_include_directories(emberfield_cli SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third-party/CLI11")
target_link_libraries(emberfield_cli PRIVATE EmberField::emberfield)
set_target_properties(emberfield_cli PROPERTIES OUTPUT_NAME emberfield)

if(EMBERFIELD_BUILD_EXAMPLES)
    add_subdirectory(example)
endif()
if(BUILD_TESTING)
    add_subdirectory(test)
endif()

find_package(Doxygen QUIET)
find_program(EMBERFIELD_CLANG_FORMAT NAMES clang-format clang-format-23 clang-format-22 clang-format-21 clang-format-20 clang-format-19 clang-format-18 clang-format-17 clang-format-16 clang-format-15)
if(EMBERFIELD_CLANG_FORMAT)
    add_custom_target(format
        COMMAND ${CMAKE_COMMAND} -E env "CLANG_FORMAT=${EMBERFIELD_CLANG_FORMAT}"
            bash "${CMAKE_CURRENT_SOURCE_DIR}/scripts/format.sh" --write
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Formatting EmberField sources" VERBATIM)
    add_custom_target(format-check
        COMMAND ${CMAKE_COMMAND} -E env "CLANG_FORMAT=${EMBERFIELD_CLANG_FORMAT}"
            bash "${CMAKE_CURRENT_SOURCE_DIR}/scripts/format.sh" --check
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Checking EmberField source formatting" VERBATIM)
endif()
if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/build/docs"
        COMMAND Doxygen::doxygen "${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Building EmberField documentation" VERBATIM)
endif()

install(TARGETS emberfield emberfield_cli EXPORT EmberFieldTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.cuh")
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/EmberField)
install(FILES third-party/picojson/LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/EmberField/picojson)
install(FILES third-party/CLI11/LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/EmberField/CLI11)
install(FILES third-party/CLI11/CLI11.hpp DESTINATION ${CMAKE_INSTALL_DATADIR}/EmberField/third-party/CLI11)
install(FILES third-party/picojson/picojson.h DESTINATION ${CMAKE_INSTALL_DATADIR}/EmberField/third-party/picojson)
install(FILES third-party/README.md DESTINATION ${CMAKE_INSTALL_DATADIR}/EmberField/third-party)
install(EXPORT EmberFieldTargets NAMESPACE EmberField:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/EmberField)
configure_package_config_file(cmake/EmberFieldConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/EmberFieldConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/EmberField)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/EmberFieldConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/EmberFieldConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/EmberFieldConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/EmberField)
