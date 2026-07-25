cmake_minimum_required(VERSION 3.20)

project(optim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find Eigen
find_package(Eigen3 REQUIRED)

# List library sources
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

# Properly scoped include paths for library
target_include_directories(optim
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include      # For library headers
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src         # Optional if needed internally
)

# Link Eigen to library
target_link_libraries(optim PUBLIC Eigen3::Eigen)

enable_testing()
add_subdirectory(tests)