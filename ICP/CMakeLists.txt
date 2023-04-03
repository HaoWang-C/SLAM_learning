cmake_minimum_required(VERSION 3.16)
project(ICP)
add_executable(test src/test.cc)
find_package(Ceres REQUIRED)
find_package(PCL REQUIRED)
include_directories(include ${CERES_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
target_link_libraries(test PRIVATE Ceres::ceres ${PCL_LIBRARIES})

