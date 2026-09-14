from __future__ import annotations

import numpy as np

from vehicle_estimation.models.nonlinear_single_track_3dof import NonlinearSingleTrack3DOF


class LongitudinalVelocityMeasurement3DOF:
    source = "sensor.longitudinal_velocity"

    def __init__(self, sigma_mps: float):
        self.sigma = float(sigma_mps)

    def predict(self, x: np.ndarray, *, delta: float) -> np.ndarray:
        return np.array([float(np.asarray(x)[0])])

    def jacobian_x(self, x: np.ndarray, *, delta: float) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0]])

    def default_covariance(self) -> np.ndarray:
        return np.array([[self.sigma**2]])


class YawRateMeasurement3DOF:
    source = "sensor.yaw_rate"

    def __init__(self, sigma_radps: float):
        self.sigma = float(sigma_radps)

    def predict(self, x: np.ndarray, *, delta: float) -> np.ndarray:
        return np.array([float(np.asarray(x)[2])])

    def jacobian_x(self, x: np.ndarray, *, delta: float) -> np.ndarray:
        return np.array([[0.0, 0.0, 1.0]])

    def default_covariance(self) -> np.ndarray:
        return np.array([[self.sigma**2]])


class RearAxleLateralAccelerationMeasurement3DOF:
    source = "sensor.lateral_acceleration_rear_axle"

    def __init__(self, model: NonlinearSingleTrack3DOF, sigma_mps2: float):
        self.model = model
        self.sigma = float(sigma_mps2)

    def predict(self, x: np.ndarray, *, delta: float) -> np.ndarray:
        ay = self.model.lateral_acceleration_rear_axle(x, delta=delta)
        return np.array([ay])

    def jacobian_x(
        self, x: np.ndarray, *, delta: float, eps: float = 1e-6
    ) -> np.ndarray:
        """Numerical measurement Jacobian dh/dx.

        Kept numerical initially so the physics stays easy to verify. It can be
        replaced by an analytical Jacobian later without changing the EKF API.
        """
        x = np.asarray(x, dtype=float).reshape(3)
        H = np.zeros((1, 3), dtype=float)
        for i in range(3):
            step = eps * max(1.0, abs(x[i]))
            xp = x.copy()
            xm = x.copy()
            xp[i] += step
            xm[i] -= step
            hp = self.model.lateral_acceleration_rear_axle(xp, delta=delta)
            hm = self.model.lateral_acceleration_rear_axle(xm, delta=delta)
            H[0, i] = (hp - hm) / (2.0 * step)
        return H

    def default_covariance(self) -> np.ndarray:
        return np.array([[self.sigma**2]])
