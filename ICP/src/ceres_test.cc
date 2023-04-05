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
using PointPair = std::pair<Eigen::Vector3d, Eigen::Vector3d>;

class SE3SophusParameterization : public ceres::LocalParameterization {
 public:
  SE3SophusParameterization() {}
  virtual ~SE3SophusParameterization() {}

  virtual bool Plus(const double *T_raw, const double *delta_raw,
                    double *T_plus_delta_raw) const override {
    // Eigen::Map is like a reference to the original data
    // T_raw is a Sophux::SE group element
    const Eigen::Map<const Sophus::SE3d> T(T_raw);
    Eigen::Map<Sophus::SE3d> T_plus_delta(T_plus_delta_raw);

    // delta_raw is a 6x1 vector
    // we need to change the sequence for updating
    const Eigen::Map<const Eigen::Vector3d> delta_phi(delta_raw);  // 先旋转部分
    const Eigen::Map<const Eigen::Vector3d> delta_rho(delta_raw +
                                                      3);  // 再平移部分

    Eigen::Matrix<double, 6, 1> delta_se3;
    delta_se3.block<3, 1>(0, 0) = delta_rho;
    delta_se3.block<3, 1>(3, 0) =
        delta_phi;  // 但进行exp()操作时，平移在前，旋转在后

    // Left multiply
    T_plus_delta = Sophus::SE3d::exp(delta_se3) * T;

    return true;
  }

  // The jacobian of Plus(x, delta) w.r.t delta at delta = 0. -> ?
  virtual bool ComputeJacobian(const double *T_raw,
                               double *jacobian_raw) const override {
    Eigen::Map<Sophus::SE3d const> T(T_raw);
    // How to define a 3 by 6 Jacobian here?
    Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> jacobian(
        jacobian_raw);
    jacobian.setZero();

    // Set the left 3 by 3 to an Identity
    // Set the right 3 by 3 to an Inverse Identity
    jacobian.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    jacobian.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
    return true;
  }

  virtual int GlobalSize() const override {
    return Sophus::SE3d::num_parameters;
  }

  virtual int LocalSize() const override { return Sophus::SE3d::DoF; }
};

class PointCostFunction : public ceres::SizedCostFunction<3, 7> {
  // size of the residual, the transform
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  PointCostFunction(const PointPair &point_pair) : point_pair_(point_pair) {}
  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const {
    // The parameters is an SE(3) transform -> len(12) array
    Eigen::Map<const Sophus::SE3d> T(parameters[0]);

    // compute the residual
    Eigen::Vector3d z = T * point_pair_.second;
    Eigen::Vector3d point_diff = point_pair_.first - z;
    residuals[0] = point_diff.x();
    residuals[0] = point_diff.y();
    residuals[0] = point_diff.z();

    // populate the jacobian matrix
    Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> jacobian(
        jacobians[0]);
    jacobian.block<3, 3>(0, 0) = Sophus::SO3d::hat(z);
    jacobian.block<3, 3>(0, 3) = Eigen::DiagonalMatrix<double, 3>(-1, -1, -1);
    return true;
  }

 private:
  // pairs of (target, source)
  PointPair point_pair_;
};

int main() { return 0; }