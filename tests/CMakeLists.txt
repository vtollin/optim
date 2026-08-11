include(FetchContent)

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
  GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(googletest)

add_executable(optim_tests
    unit/steplengthpolicy/ArmijoUnit.cpp
    unit/steplengthpolicy/StrongWolfeUnit.cpp
    unit/functions/RosenbrockUnit.cpp
    unit/functions/GeneralizedRosenbrockUnit.cpp
    unit/functions/HimmelblauUnit.cpp
    unit/functions/WoodUnit.cpp
    unit/base/LineSearchBaseUnit.cpp
    unit/base/TrustRegionBaseUnit.cpp
    integration/linesearch/SteepestDescentIntegration.cpp
    integration/linesearch/NewtonIntegration.cpp
    integration/linesearch/BFGSIntegration.cpp
    integration/trustregion/DoglegIntegration.cpp
    integration/trustregion/MoreSorensenIntegration.cpp
    integration/trustregion/SteihaugCGIntegration.cpp
)

target_link_libraries(optim_tests PRIVATE optim optim_test_functions gtest_main)

include(GoogleTest)
gtest_discover_tests(optim_tests)