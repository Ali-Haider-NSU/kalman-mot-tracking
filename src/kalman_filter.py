"""Linear Kalman filter and the constant-velocity motion model."""
import numpy as np


def kf1d_step(x, P, z, q, r):
    """One cycle of a 1D Kalman filter with a constant-position model.

    Returns the new estimate, its variance and the Kalman gain.
    """
    x_pred, P_pred = x, P + q            # predict: the state may have drifted by q
    K = P_pred / (P_pred + r)            # Kalman gain
    x_new = x_pred + K * (z - x_pred)    # update with the innovation z - x_pred
    P_new = (1 - K) * P_pred
    return x_new, P_new, K


class KalmanFilter:
    """Linear Kalman filter: x' = F x + w,  z = H x + v,  w ~ N(0, Q),  v ~ N(0, R)."""

    def __init__(self, F, H, Q, R, x0, P0):
        self.F, self.H, self.Q, self.R = F, H, Q, R
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        """Correct the state with measurement z; returns the innovation y and its covariance S."""
        y = np.asarray(z, dtype=float) - self.H @ self.x      # innovation
        S = self.H @ self.P @ self.H.T + self.R               # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)              # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
        return y, S

    def nis(self, y, S):
        """Normalised innovation squared: about dim(z) on average if Q and R are well tuned."""
        return float(y @ np.linalg.inv(S) @ y)


def constant_velocity_model(dt, q, r):
    """State [x, y, vx, vy], measurement [x, y].

    q: standard deviation of the (white-noise) acceleration, r: measurement standard deviation.
    """
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    G = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
    Q = q**2 * G @ G.T
    R = r**2 * np.eye(2)
    return F, H, Q, R
