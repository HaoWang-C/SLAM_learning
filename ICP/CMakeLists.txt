cmake_minimum_required(VERSION 3.16)
project(ICP)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

add_executable(pcl_visul src/pcl_visul.cc)
add_executable(ceres_test src/ceres_test.cc)
add_executable(try_icp src/try_icp.cc)


target_link_libraries(pcl_visul PRIVATE ${CERES_LIBRARIES}  ${PCL_LIBRARIES})
target_link_libraries(ceres_test PRIVATE  ${CERES_LIBRARIES} ${Sophus_LIBRARIES} fmt)
target_link_libraries(try_icp PRIVATE ${CERES_LIBRARIES} ${PCL_LIBRARIES})



