#include "ceres/ceres.h"
#include "glog/logging.h"
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  
  return 0;
}