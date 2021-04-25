cmake_minimum_required(VERSION 3.13)
project(min_cut LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp")
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(TBB REQUIRED)

SET(LIGMG_DIR "${CMAKE_SOURCE_DIR}/dependencies/ligmg/build/LIGMG")
find_package(LIGMG REQUIRED)

set(CUSP_INCLUDES "dependencies/cusplibrary")
set(THRUST_INCLUDES "dependencies/thrust")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --cudart; static; -O4 ;
                    --expt-relaxed-constexpr;)
cuda_add_library(cusp_cg cusp_cg.cu)
set_property(TARGET cusp_cg PROPERTY CUDA_ARCHITECTURES 50)
target_link_libraries(cusp_cg Eigen3::Eigen tbb)
target_include_directories(cusp_cg PUBLIC ${CUSP_INCLUDES} ${THRUST_INCLUDES})

add_executable(main main.cpp)
# target_include_directories(main PUBLIC )

target_link_libraries(main Eigen3::Eigen cusp_cg tbb ligmg::LIGMG)
