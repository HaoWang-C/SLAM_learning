#include <pcl/common/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/cloud_viewer.h>

#include <iostream>

int main() {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_1(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2(
      new pcl::PointCloud<pcl::PointXYZ>);

  pcl::io::loadPCDFile(
      "/home/mars09/Desktop/learningSLAM/src/SLAM_learning/ICP/test/bun01.pcd",
      *cloud_1);
  pcl::io::loadPCDFile(
      "/home/mars09/Desktop/learningSLAM/src/SLAM_learning/ICP/test/bun02.pcd",
      *cloud_2);

  // store the target points into a K-D tree
  // find the neaest neighbour for each source points in the tree
  // form the association -> {i, j}

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

  kdtree.setInputCloud(cloud_1);
  pcl::PointXYZ searchPoint;
  searchPoint.x = 0.03;
  searchPoint.y = 0.1;
  searchPoint.z = 0;
  int K = 10;

  std::vector<int> pointIdxKNNSearch(K);
  std::vector<float> pointKNNSquaredDistance(K);
  std::cout << "K nearest neighbor search at (" << searchPoint.x << " "
            << searchPoint.y << " " << searchPoint.z << ") with K=" << K
            << std::endl;
  if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch,
                            pointKNNSquaredDistance) > 0) {
    for (std::size_t i = 0; i < pointIdxKNNSearch.size(); ++i)
      std::cout << "    " << (*cloud_1)[pointIdxKNNSearch[i]].x << " "
                << (*cloud_1)[pointIdxKNNSearch[i]].y << " "
                << (*cloud_1)[pointIdxKNNSearch[i]].z
                << " (squared distance: " << pointKNNSquaredDistance[i] << ")"
                << std::endl;
  }
  return 0;
}

// // Neighbors within radius search
// std::vector<int> pointIdxRadiusSearch;
// std::vector<float> pointRadiusSquaredDistance;
// float radius = 256.0f * rand () / (RAND_MAX + 1.0f);
// if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch,
// pointRadiusSquaredDistance) > 0 )
// {
//   for (std::size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
//     std::cout << "    "  <<   (*cloud)[ pointIdxRadiusSearch[i] ].x
//               << " " << (*cloud)[ pointIdxRadiusSearch[i] ].y
//               << " " << (*cloud)[ pointIdxRadiusSearch[i] ].z
//               << " (squared distance: " << pointRadiusSquaredDistance[i] <<
//               ")" << std::endl;
// }
