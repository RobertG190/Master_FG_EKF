from __future__ import annotations

import numpy as np

from .base import MeasurementModel


class YawRateMeasurement(MeasurementModel):
    source = "sensor.yaw_rate"

    def __init__(self, sigma_radps: float):
        self.sigma = float(sigma_radps)

    def predict(self, x: np.ndarray, *, delta: float, vx: float) -> np.ndarray:
        return np.array([float(x[1])])

    def jacobian_x(self, x: np.ndarray, *, delta: float, vx: float) -> np.ndarray:
        return np.array([[0.0, 1.0]])

    def default_covariance(self) -> np.ndarray:
        return np.array([[self.sigma**2]])
