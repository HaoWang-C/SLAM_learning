#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include "ceres/ceres.h"
#include "ceres/local_parameterization.h"
#include "ceres/solver.h"
#include "glog/logging.h"
#include "iostream"
#include "sophus/se3.hpp"
#include <Eigen/Core>

#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/cloud_viewer.h>
#include "registration_solve.h"

using SE3SophusParameterization = registration::SE3SophusParameterization;
using PointCostFunction = registration::PointCostFunction;

namespace registration {
    class RegistrationProblem {
    public:
        int num_iteration_ = 0;
        void Setup(std::string target_path) {
            printf("\nLoading target points...\n");
            target_points_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(
                new pcl::PointCloud<pcl::PointXYZ>());
            pcl::io::loadPCDFile(target_path, *target_points_);

            // Apply transformation to create a source points
            // {x,y,z,angx,angy,angz}
            double ground_truth_vector[6] = { 0.5,0,0,0,0,0 };
            gt_transform_ = Sophus::Vector<double, 6>(ground_truth_vector);

            // Print the transformation
            printf("\nGround truth transformation\n");
            std::cout << Sophus::SE3d::exp(gt_transform_).matrix() << std::endl;

            // Apply transfomation
            // Executing the transformation
            printf("\nCreating source points...\n");
            source_points_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(
                new pcl::PointCloud<pcl::PointXYZ>());
            pcl::transformPointCloud(*target_points_, *source_points_, Sophus::SE3d::exp(gt_transform_).matrix());

            // Build a KD tree for the target points
            printf("\nBuilding KD-Tree for target points...\n");
            target_kdtree_.setInputCloud(target_points_);
        }

        void Visulise() {
            // Visualization
            printf("\nVisualising, press Q to quit...\n");
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
            printf("\nFinding association...\n");
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
                        PointPair(target, source));
                }
            }
            double association_rate =
                associations_points_.size() / target_points_->size();
            printf("\nAssociation rate\n");
            std::cout << "rate: " << association_rate << " = "
                << associations_points_.size() << " / " << target_points_->size()
                << std::endl;
        }

        void SolveAndUpdate() {
            Problem problem;
            // the pose is in se(3) -> 6 param
            Eigen::Map<Sophus::Vector<double, 6>> iteration_solve(solve_pose_);
            std::cout << "Initial solve: \n" << Sophus::SE3d::exp(iteration_solve).matrix() << std::endl;

            for (const auto& point : associations_points_) {
                ceres::CostFunction* costfunc = new PointCostFunction(point);
                problem.AddResidualBlock(costfunc, nullptr, solve_pose_);
                problem.SetParameterization(solve_pose_, new SE3SophusParameterization());
            }

            Solver::Options options;
            options.max_num_iterations = 25;
            options.linear_solver_type = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = false;
            Solver::Summary summary;
            Solve(options, &problem, &summary);

            std::cout << "Final solve: \n" << Sophus::SE3d::exp(iteration_solve).matrix() << std::endl;

            // Update the source_points_ using the iteration solution
            // pcl::PointCloud<pcl::PointXYZ> transformed_source_point_cloud;
            pcl::transformPointCloud(*source_points_, *source_points_, Sophus::SE3d::exp(iteration_solve).matrix());
            // source_points_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(transformed_source_point_cloud);

            // Clear the previous association
            associations_points_.clear();
            num_iteration_++;
        }

    private:
        pcl::PointCloud<pcl::PointXYZ>::Ptr target_points_;
        pcl::PointCloud<pcl::PointXYZ>::Ptr source_points_;
        Sophus::Vector<double, 6> gt_transform_;

        pcl::KdTreeFLANN<pcl::PointXYZ> target_kdtree_;
        int num_neighbour_ = 10;
        // pairs of (target, source)
        std::vector<PointPair> associations_points_;

        // {x,y,z,angx,angy,angz}
        double solve_pose_[6] = { 0,0,0,0,0,0 };
    };
}
