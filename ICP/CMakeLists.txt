cmake_minimum_required(VERSION 3.16)
project(ICP)

set(Qt5_DIR /opt/homebrew/Cellar/qt@5/5.15.8_2/lib/cmake/Qt5)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

include_directories(include)

add_executable(registration src/registration.cc)
target_link_libraries(registration PRIVATE  ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${Sophus_LIBRARIES} fmt::fmt)



