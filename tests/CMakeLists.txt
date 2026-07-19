include(FetchContent)

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)

FetchContent_MakeAvailable(googletest)

add_executable(optim_tests
    unit/steplengthpolicy/ArmijoUnit.cpp
    unit/steplengthpolicy/StrongWolfeUnit.cpp
    unit/functions/BealeUnit.cpp
    unit/functions/RosenbrockUnit.cpp
    unit/functions/HimmelblauUnit.cpp
    unit/functions/WoodUnit.cpp
    unit/functions/LogisticRegressionLossUnit.cpp
    integration/linesearch/SteepestDescentIntegration.cpp
    integration/linesearch/NewtonIntegration.cpp
    integration/linesearch/BFGSIntegration.cpp
    integration/trustregion/DoglegIntegration.cpp
    integration/trustregion/MoreSorensenIntegration.cpp
    integration/trustregion/SteihaugCGIntegration.cpp
)

target_include_directories(optim_tests PRIVATE ${PROJECT_SOURCE_DIR}/test_functions)

target_link_libraries(optim_tests
    PRIVATE
        optim
        gtest_main
)

include(GoogleTest)
gtest_discover_tests(optim_tests)