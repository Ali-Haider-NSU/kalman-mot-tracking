// Constant-velocity tracking of one point: the C++ version of the Python example.
#include <cmath>
#include <iostream>
#include <random>
#include "kalman_filter.hpp"

int main()
{
    constexpr double dt = 1.0 / 30.0, q = 150.0, r = 4.0;
    using KF = KalmanFilter<4, 2>;

    KF::StateMat F = KF::StateMat::Identity();
    F(0, 2) = dt;  F(1, 3) = dt;
    KF::ObsMat H = KF::ObsMat::Zero();
    H(0, 0) = 1;  H(1, 1) = 1;
    Eigen::Matrix<double, 4, 2> G;
    G << dt * dt / 2, 0,  0, dt * dt / 2,  dt, 0,  0, dt;
    KF::StateMat Q = q * q * G * G.transpose();
    KF::MeasMat R = r * r * KF::MeasMat::Identity();

    std::mt19937 gen(0);
    std::normal_distribution<double> noise(0.0, r);

    // truth: constant velocity (120, 40) px/s starting at (200, 300)
    KF::StateVec x0;  x0 << 200 + noise(gen), 300 + noise(gen), 0, 0;
    KF::StateMat P0 = KF::StateMat::Zero();
    P0.diagonal() << 16, 16, 1e4, 1e4;
    KF kf(F, H, Q, R, x0, P0);

    double se_raw = 0, se_kf = 0, nis_sum = 0;
    const int n = 150;
    for (int k = 1; k < n; ++k) {
        const double tx = 200 + 120 * k * dt, ty = 300 + 40 * k * dt;
        KF::MeasVec z;  z << tx + noise(gen), ty + noise(gen);
        kf.predict();
        nis_sum += kf.update(z);
        if (k >= 30) {                                   // ignore the start-up phase
            se_raw += std::pow(z(0) - tx, 2) + std::pow(z(1) - ty, 2);
            se_kf  += std::pow(kf.state()(0) - tx, 2) + std::pow(kf.state()(1) - ty, 2);
        }
    }
    const int m = n - 30;
    std::cout << "RMSE raw: " << std::sqrt(se_raw / m)
              << " px, Kalman: " << std::sqrt(se_kf / m) << " px\n";
    std::cout << "velocity estimate: " << kf.state()(2) << ", " << kf.state()(3)
              << " px/s (true 120, 40)\n";
    std::cout << "mean NIS: " << nis_sum / (n - 1) << " (ideal 2)\n";
    return 0;
}
