cmake_minimum_required(VERSION 3.13)
project(min_cut LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp -ffast-math")
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(TBB REQUIRED)

add_executable(main main.cpp)

target_link_libraries(main Eigen3::Eigen tbb)
