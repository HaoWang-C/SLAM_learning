#include "ceres/ceres.h"

#include <Eigen/Core>

#include "ceres/local_parameterization.h"
#include "glog/logging.h"
#include "sophus/se3.hpp"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

struct PointCloudResidual {
  PointCloudResidual(double x, double y, double z) : x_(x), y_(y), z_(z) {}
  template <typename T>
  bool operator()(const T *const x, const T *const y, const T *const z,
                  T *residual) const {
    residual[0] = x_ - x[0];
    residual[1] = y_ - y[0];
    residual[2] = z_ - z[0];
    return true;
  }

 private:
  const double x_;
  const double y_;
  const double z_;
};

// int main(int argc, char** argv) {
//   google::InitGoogleLogging(argv[0]);
//   Problem problem;
//   Eigen::Vector3d Point_1(1, 1, 1);
//   Eigen::Vector3d Point_2(-2, -2, -2);
//   std::vector<Eigen::Vector3d> points;
//   points.push_back(Point_1);
//   points.push_back(Point_2);

//   Eigen::Vector3d initial_guess(0, 0, 0);

//   for (const auto& point : points) {
//     problem.AddResidualBlock(
//         new AutoDiffCostFunction<xyzResidual, 3, 1, 1, 1>(
//             new xyzResidual(point.x(), point.y(), point.z())),
//         nullptr, &initial_guess.x(), &initial_guess.y(), &initial_guess.z());
//   }
//   Solver::Options options;
//   options.max_num_iterations = 25;
//   options.linear_solver_type = ceres::DENSE_QR;
//   options.minimizer_progress_to_stdout = true;
//   Solver::Summary summary;
//   Solve(options, &problem, &summary);
//   std::cout << summary.BriefReport() << "\n";
//   //   std::cout << "Initial m: " << 0.0 << " c: " << 0.0 << "\n";
//   std::cout << "Final   x: " << initial_guess.x() << " y: " <<
//   initial_guess.y()
//             << " z: " << initial_guess.z() << "\n";
//   return 0;
// }

class SE3SophusParameterization : public ceres::LocalParameterization {
 public:
  virtual ~SE3SophusParameterization() {}

  virtual bool Plus(const double *x, const double *delta,
                    double *x_plus_delta) const override {
    // 转化为Eigen变量
    const Eigen::Map<const Sophus::SE3d> T(x);
    Eigen::Map<Sophus::SE3d> T_plus_delta(x_plus_delta);
    const Eigen::Map<const Eigen::Vector3d> delta_phi(delta);  // 先旋转部分
    const Eigen::Map<const Eigen::Vector3d> delta_rho(delta + 3);  // 再平移部分

    Eigen::Matrix<double, 6, 1> delta_se3;
    delta_se3.block<3, 1>(0, 0) = delta_rho;
    delta_se3.block<3, 1>(3, 0) =
        delta_phi;  // 但进行exp()操作时，平移在前，旋转在后

    // 左乘更新
    T_plus_delta = Sophus::SE3d::exp(delta_se3) * T;

    return true;
  }

  virtual bool ComputeJacobian(const double *x,
                               double *jacobian) const override {
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > jaco_tmp(
        jacobian);
    jaco_tmp.setZero();
    jaco_tmp.topRows(6).setIdentity();
    return true;
  }

  virtual int GlobalSize() const override {
    return Sophus::SE3d::num_parameters;
  }

  virtual int LocalSize() const override { return Sophus::SE3d::DoF; }
};

int main() { return 0; }