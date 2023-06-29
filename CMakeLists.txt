cmake_minimum_required(VERSION 3.5)
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/cmake/TBB/")
find_package(TBB REQUIRED)
project(projekt) # this does not matter for our simple case
add_executable(solver_Eigen_block src/solver_Eigen_block.cpp) # this determines the executable name
target_include_directories(solver_Eigen_block PUBLIC include)
target_compile_features(solver_Eigen_block PUBLIC cxx_std_20) # we need C++20
find_package(benchmark REQUIRED)
target_link_libraries(solver_Eigen_block PUBLIC TBB::tbb benchmark::benchmark)