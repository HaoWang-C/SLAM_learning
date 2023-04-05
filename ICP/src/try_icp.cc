#include <ceres/local_parameterization.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/cloud_viewer.h>

#include <Eigen/Core>
#include <iostream>
#include <sophus/se3.hpp>

#include "ceres/ceres.h"
#include "glog/logging.h"

class RegistrationProblem {
 public:
  void Setup() {
    // Creat target points
    target_points_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(
        new pcl::PointCloud<pcl::PointXYZ>());
    pcl::io::loadPCDFile(
        "/home/mars09/Desktop/learningSLAM/src/SLAM_learning/ICP/test/"
        "bun01.pcd",
        *target_points_);

    // Apply transformation to create a source points
    Eigen::Affine3f gt_transform_ = Eigen::Affine3f::Identity();
    // Define a translation of 1 meters on the x axis.
    gt_transform_.translation() << 0.3, 0.3, 0.3;
    float theta = M_PI / 4;  // The angle of rotation in radians
    // The same rotation matrix as before; theta radians around Z axis
    gt_transform_.rotate(Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitZ()));

    // Print the transformation
    printf("\nGround truth transformation\n");
    std::cout << gt_transform_.matrix() << std::endl;

    // Apply transfomation
    // Executing the transformation
    source_points_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(
        new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*target_points_, *source_points_, gt_transform_);

    // Build a KD tree for the target points
    target_kdtree_.setInputCloud(target_points_);
  }

  void Visulise() {
    // Visualization
    printf(
        "\nPoint cloud colors :  white  = target points\n"
        "                        red  = source points\n");
    pcl::visualization::PCLVisualizer viewer("Problem Viewer");

    // Define R,G,B colors for the point cloud
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        target_cloud_color_handler(target_points_, 255, 255, 255);
    viewer.addPointCloud(target_points_, target_cloud_color_handler,
                         "target_points_");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        source_cloud_color_handler(source_points_, 230, 20, 20);  // Red
    viewer.addPointCloud(source_points_, source_cloud_color_handler,
                         "source_points_");

    viewer.addCoordinateSystem(1.0, "cloud", 0);
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0);
    // Setting background to a dark grey
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target_points_");
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "source_points_");
    viewer.setPosition(800, 400);  // Setting visualiser window position

    while (!viewer.wasStopped()) {
      // Display the visualiser until 'q' key is pressed
      viewer.spinOnce();
    }
  }

  void FindAssociation() {
    std::vector<int> pointIdxKNNSearch(num_neighbour_);
    std::vector<float> pointKNNSquaredDistance(num_neighbour_);
    for (const auto& source_point : *source_points_) {
      if (target_kdtree_.nearestKSearch(source_point, num_neighbour_,
                                        pointIdxKNNSearch,
                                        pointKNNSquaredDistance) > 0) {
        // Found association
        Eigen::Vector3d target((*target_points_)[pointIdxKNNSearch[0]].x,
                               (*target_points_)[pointIdxKNNSearch[0]].y,
                               (*target_points_)[pointIdxKNNSearch[0]].z);
        Eigen::Vector3d source(source_point.x, source_point.y, source_point.z);
        associations_points_.push_back(
            std::pair<Eigen::Vector3d, Eigen::Vector3d>(target, source));
      }
    }
    double association_rate =
        associations_points_.size() / target_points_->size();
    printf("\nAssociation rate\n");
    std::cout << "rate: " << association_rate << " = "
              << associations_points_.size() << " / " << target_points_->size()
              << std::endl;
  }

 private:
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_points_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_points_;
  Eigen::Affine3f gt_transform_;

  pcl::KdTreeFLANN<pcl::PointXYZ> target_kdtree_;
  int num_neighbour_ = 10;
  // pairs of (target, source)
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> associations_points_;
};

int main() {
  RegistrationProblem registrationproblem;
  registrationproblem.Setup();
  registrationproblem.Visulise();
  registrationproblem.FindAssociation();

  // Construct a KD tree for target points
  return 0;
}