include(FetchContent)
FetchContent_Declare(benchmark 
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
    GIT_SHALLOW TRUE)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(benchmark)

add_executable(optim_bench bench_main.cpp)
target_link_libraries(optim_bench PRIVATE optim optim_test_functions benchmark::benchmark_main)
target_compile_options(optim_bench PRIVATE -fno-omit-frame-pointer)