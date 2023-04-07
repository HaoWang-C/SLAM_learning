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

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
using PointPair = std::pair<Eigen::Vector3d, Eigen::Vector3d>;

namespace registration {
    class SE3SophusParameterization: public ceres::LocalParameterization {
    public:
        SE3SophusParameterization() {}
        virtual ~SE3SophusParameterization() {}
        // x_se3 is in se(3)
        // dleta_x is in se(3)
        // x_plus_delta_x is back in se(3)
        virtual bool Plus(const double* x, const double* delta_x,
            double* x_plus_delta) const override {
            // Eigen::Map is like a reference to the original data
            // x is a double [6] which represent a Sophus::Vector<double, 6>
            const Eigen::Map<const Sophus::Vector<double, 6>> x_se3(x);
            Sophus::SE3d x_SE3 = Sophus::SE3d::exp(x_se3);

            // dleta_x is also a double [6] which represent a Sophus::Vector<double, 6>
            const Eigen::Map<const Sophus::Vector<double, 6>> delta_x_se3(delta_x);

            Eigen::Map< Sophus::Vector<double, 6>> x_plus_delta_se3(x_plus_delta);

            // Left multiply
            Sophus::SE3d T_plus_delta = Sophus::SE3d::exp(delta_x_se3) * x_SE3;
            x_plus_delta_se3 = T_plus_delta.log();
            return true;
        }

        // The jacobian of Plus(x, delta) w.r.t delta at delta = 0
        // note that x is in se(3)
        virtual bool ComputeJacobian(const double* x,
            double* jacobian) const override {
            ceres::MatrixRef(jacobian, 6, 6) = ceres::Matrix::Identity(6, 6);
            return true;
        }

        virtual int GlobalSize() const override {
            return Sophus::SE3d::DoF;
        }

        virtual int LocalSize() const override { return Sophus::SE3d::DoF; }
    };

    class PointCostFunction: public ceres::SizedCostFunction<3, 6> {
        // size of the residual, the transform in se(3)
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        PointCostFunction(const PointPair& point_pair): point_pair_(point_pair) {}
        virtual bool Evaluate(double const* const* parameters, double* residuals,
            double** jacobians) const {
            // The parameters is an se(3)
            // change it to SE(3) now
            Eigen::Map <const Sophus::Vector<double, 6>> T_se3(parameters[0]);
            Sophus::SE3d T = Sophus::SE3d::exp(T_se3);
            Eigen::Vector3d z = T * point_pair_.second;
            Eigen::Vector3d point_diff = point_pair_.first - z;
            residuals[0] = point_diff.x();
            residuals[1] = point_diff.y();
            residuals[2] = point_diff.z();

            // populate the jacobian matrix 3 by 6
            if (jacobians) {
                if (jacobians[0]) {
                    Eigen::Matrix<double, 3, 6> jaco;
                    jaco.block(0, 0, 3, 3) = Eigen::DiagonalMatrix<double, 3, 3>(-1.0, -1.0, -1.0);
                    jaco.block(0, 3, 3, 3) = Sophus::SO3d::hat(z);
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
}