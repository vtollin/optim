cmake_minimum_required(VERSION 3.20)

project(optim)


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(Eigen3 REQUIRED)

set(CORE_SOURCES
    src/optimizer/OptimizerBase.cpp
)

set(TRUSTREGION_SOURCES
    src/trustregion/Dogleg.cpp
    src/trustregion/TrustRegionBase.cpp
    src/trustregion/MoreSorensen.cpp
    src/trustregion/SteihaugCG.cpp
)

set(LINESEARCH_SOURCES 
    src/linesearch/ArmijoBacktracking.cpp
    src/linesearch/StrongWolfe.cpp
    src/linesearch/BFGS.cpp
    src/linesearch/SteepestDescent.cpp
    src/linesearch/Newton.cpp
    src/linesearch/LineSearchBase.cpp
    src/linesearch/StepLengthMethod.cpp
)

set(DIAGNOSTIC_SOURCES
    src/diagnostics/GradientChecker.cpp
    src/diagnostics/HessianChecker.cpp
)

add_library(optim STATIC ${CORE_SOURCES} ${TRUSTREGION_SOURCES} ${LINESEARCH_SOURCES} ${DIAGNOSTIC_SOURCES})

target_compile_features(optim PUBLIC cxx_std_17)

target_include_directories(optim
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(optim PUBLIC Eigen3::Eigen)

enable_testing()
add_subdirectory(tests)