cmake_minimum_required(VERSION 3.13)
project(min_cut LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/home/little/Libs/eigen/cmake/")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp")

find_package(SPQR)
find_package(CHOLMOD)
find_package(UMFPACK)
find_package(Eigen3 REQUIRED)

add_executable(main main.cpp)
message(${SPQR_INCLUDES})
target_include_directories(main PUBLIC ${SPQR_INCLUDES} ${CHOLMOD_INCLUDES})

target_link_libraries(main Eigen3::Eigen ${SPQR_LIBRARIES})
