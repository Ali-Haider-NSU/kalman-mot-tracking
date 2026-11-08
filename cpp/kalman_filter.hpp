#pragma once
#include <Eigen/Dense>

// Linear Kalman filter with fixed sizes known at compile time (fast, no heap allocation).
// NX: state dimension, NZ: measurement dimension.
template <int NX, int NZ>
class KalmanFilter
{
public:
    using StateVec = Eigen::Matrix<double, NX, 1>;
    using StateMat = Eigen::Matrix<double, NX, NX>;
    using MeasVec  = Eigen::Matrix<double, NZ, 1>;
    using MeasMat  = Eigen::Matrix<double, NZ, NZ>;
    using ObsMat   = Eigen::Matrix<double, NZ, NX>;

    KalmanFilter(const StateMat &F, const ObsMat &H, const StateMat &Q, const MeasMat &R,
                 const StateVec &x0, const StateMat &P0)
        : F_(F), H_(H), Q_(Q), R_(R), x_(x0), P_(P0) {}

    void predict()
    {
        x_ = F_ * x_;
        P_ = F_ * P_ * F_.transpose() + Q_;
    }

    // returns the normalised innovation squared (NIS) for consistency checks
    double update(const MeasVec &z)
    {
        const MeasVec y = z - H_ * x_;                                   // innovation
        // innovation covariance
        const MeasMat S = H_ * P_ * H_.transpose() + R_;
        const Eigen::Matrix<double, NX, NZ> K = P_ * H_.transpose() * S.inverse();
        x_ += K * y;
        P_ = (StateMat::Identity() - K * H_) * P_;
        return y.dot(S.ldlt().solve(y));
    }

    const StateVec &state() const { return x_; }
    const StateMat &covariance() const { return P_; }

private:
    StateMat F_;
    ObsMat H_;
    StateMat Q_;
    MeasMat R_;
    StateVec x_;
    StateMat P_;
};
