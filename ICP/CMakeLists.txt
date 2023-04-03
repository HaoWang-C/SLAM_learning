cmake_minimum_required (VERSION 3.5)

project(ICP)
set(gflags_DIR "C:\\Users\\hwang\\OneDrive\\Desktop\\working_repo\\src\\vcpkg\\installed\\x86-windows\\share\\gflags")
set(glog_DIR "C:\\Users\\hwang\\OneDrive\\Desktop\\working_repo\\src\\vcpkg\\installed\\x86-windows\\share\\glog")
set(Eigen3_DIR "C:\\Users\\hwang\\OneDrive\\Desktop\\working_repo\\src\\vcpkg\\installed\\x86-windows\\share\\eigen3")
set(Ceres_DIR "C:\\Users\\hwang\\OneDrive\\Desktop\\working_repo\\src\\vcpkg\\installed\\x86-windows\\share\\ceres")

add_executable(test src/test.cc)
find_package(Ceres REQUIRED)
target_link_libraries(test PRIVATE Ceres)