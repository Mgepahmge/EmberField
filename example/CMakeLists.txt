add_executable(example-1 example-1.cu)
target_link_libraries(example-1 PRIVATE ThermoFEM::thermofem)

add_executable(example-2 example-2.cu)
target_link_libraries(example-2 PRIVATE ThermoFEM::thermofem)

add_executable(example-3 example-3.cu)
target_link_libraries(example-3 PRIVATE ThermoFEM::thermofem)

add_executable(example-4 example-4.cu)
target_link_libraries(example-4 PRIVATE ThermoFEM::thermofem)

add_executable(mesh-roundtrip mesh-roundtrip.cu)
target_link_libraries(mesh-roundtrip PRIVATE ThermoFEM::thermofem)

add_executable(case-inspect case-inspect.cu)
target_link_libraries(case-inspect PRIVATE ThermoFEM::thermofem)
if(BUILD_TESTING)
    add_test(NAME CaseSteadyIntegration COMMAND case-inspect "${CMAKE_CURRENT_SOURCE_DIR}/cases/steady.json")
    add_test(NAME CaseTransientIntegration COMMAND case-inspect "${CMAKE_CURRENT_SOURCE_DIR}/cases/transient.json")
endif()

add_executable(solve-matrix solve-matrix.cu)
target_link_libraries(solve-matrix PRIVATE ThermoFEM::thermofem)
if(BUILD_TESTING)
    add_test(NAME MatrixMarketIntegration
        COMMAND solve-matrix "${PROJECT_SOURCE_DIR}/test/matrix-market/poisson.mtx" amg)
endif()
