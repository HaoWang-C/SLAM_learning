#include "ceres/ceres.h"
#include "ceres/local_parameterization.h"
#include "ceres/solver.h"
#include "glog/logging.h"
#include "iostream"
#include "sophus/se3.hpp"
#include <Eigen/Core>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
using PointPair = std::pair<Eigen::Vector3d, Eigen::Vector3d>;

class SE3SophusParameterization: public ceres::LocalParameterization {
public:
  SE3SophusParameterization() {}
  virtual ~SE3SophusParameterization() {}

  virtual bool Plus(const double* T_raw, const double* delta_raw,
    double* T_plus_delta_raw) const override {
    // Eigen::Map is like a reference to the original data
    // T_raw is a Sophux::SE group element
    const Eigen::Map<const Sophus::SE3d> T(T_raw);
    Eigen::Map<Sophus::SE3d> T_plus_delta(T_plus_delta_raw);

    // delta_raw is a 6x1 vector
    // we need to change the sequence for updating
    const Eigen::Map<const Eigen::Vector3d> delta_phi(delta_raw); // 先旋转部分
    const Eigen::Map<const Eigen::Vector3d> delta_rho(delta_raw +
      3); // 再平移部分

    Eigen::Matrix<double, 6, 1> delta_se3;
    delta_se3.block<3, 1>(0, 0) = delta_rho;
    delta_se3.block<3, 1>(3, 0) = delta_phi; // 但进行exp()操作时，平移在前，旋转在后

    // Left multiply
    T_plus_delta = Sophus::SE3d::exp(delta_se3) * T;
    return true;
  }

  // The jacobian of Plus(x, delta) w.r.t delta at delta = 0. -> ?
  virtual bool ComputeJacobian(const double* T_raw,
    double* jacobian_raw) const override {
    Eigen::Map<Sophus::SE3d const> T(T_raw);
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> jacobian(
      jacobian_raw);
    jacobian.setZero();

    // Set the left 3 by 3 to an Identity
    // Set the right 3 by 3 to an Inverse Identity
    jacobian.block<3, 3>(4, 0) = Eigen::Matrix3d::Identity();
    jacobian.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
    return true;
  }

  virtual int GlobalSize() const override {
    return Sophus::SE3d::num_parameters;
  }

  virtual int LocalSize() const override { return Sophus::SE3d::DoF; }
};

class PointCostFunction: public ceres::SizedCostFunction<3, 7> {
  // size of the residual, the transform
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  PointCostFunction(const PointPair& point_pair): point_pair_(point_pair) {}
  virtual bool Evaluate(double const* const* parameters, double* residuals,
    double** jacobians) const {
    // The parameters is an SE(3) transform -> len(7) array
    Eigen::Map<const Sophus::SE3d> T(parameters[0]);

    std::cout << "compute the residual" << std::endl;
    Eigen::Vector3d z = T * point_pair_.second;
    Eigen::Vector3d point_diff = point_pair_.first - z;
    residuals[0] = point_diff.x();
    residuals[1] = point_diff.y();
    residuals[2] = point_diff.z();
    std::cout << "compute the jacobian" << std::endl;;
    // populate the jacobian matrix
    if (jacobians) {
      if (jacobians[0]) {
        // Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> jacobian(
        //   jacobians[0]);
        // Eigen::Matrix<double, 3, 6, Eigen::RowMajor> jacobian_transpose;
        // std::cout << "jacobian_before: \n" << jacobian_transpose << std::endl;
        // jacobian_transpose.block<3, 3>(0, 0) = Sophus::SO3d::hat(z);
        // jacobian_transpose.block<3, 3>(0, 3) = Eigen::DiagonalMatrix<double, 3, 3>(-1.0, -1.0, -1.0);
        // std::cout << "jacobian_after: \n" << jacobian_transpose << std::endl;
        // jacobian = jacobian_transpose.transpose();
        // std::cout << "j_T \n" << jacobian << std::endl;
        Eigen::Matrix<double, 3, 6> jaco;
        jaco.block(0, 0, 3, 3) = Eigen::DiagonalMatrix<double, 3, 3>(-1.0, -1.0, -1.0);
        jaco.block(0, 3, 3, 3) = Sophus::SO3d::hat(z);
        std::cout << "jaco : \n" << jaco << std::endl;
        Eigen::Matrix<double, 6, 3 > jaco_transpose = jaco.transpose();
        for (int i = 0; i < 18;++i) {
          jacobians[0][i] = jaco_transpose(i);
        }
      }
    }
    return true;
  }

private:
  // pairs of (target, source)
  PointPair point_pair_;
};

int main() {
  // setup two points
  std::vector<PointPair> associated_points;
  Eigen::Vector3d target_1{1, 1, 0};
  Eigen::Vector3d target_2{2, 2, 0};
  Eigen::Vector3d target_3{3, 3, 0};

  Eigen::Vector3d source_1{2, 1, 0};
  Eigen::Vector3d source_2{3, 2, 0};
  Eigen::Vector3d source_3{4, 3, 0};
  associated_points.push_back(PointPair(target_1, source_1));
  associated_points.push_back(PointPair(target_2, source_2));
  associated_points.push_back(PointPair(target_3, source_3));

  ceres::LocalParameterization* local_param = new SE3SophusParameterization();
  Problem problem;

  // Need a function to convert the ground truth transforms to -> a doduble array
  double solve_pose[7] = { 0,0,0,0,0,0,0 };
  // for (auto& associated_point : associated_points) {
  //   ceres::CostFunction* costfunc = new PointCostFunction(associated_point);
  //   problem.AddResidualBlock(costfunc, nullptr, solve_pose);
  //   problem.SetParameterization(solve_pose, local_param);
  // }

  PointPair point = PointPair(target_1, source_1);
  ceres::CostFunction* costfunc = new PointCostFunction(point);
  problem.AddResidualBlock(costfunc, nullptr, solve_pose);
  problem.SetParameterization(solve_pose, local_param);

  Solver::Options options;
  options.max_num_iterations = 25;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  return 0;
}