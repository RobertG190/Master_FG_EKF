from __future__ import annotations

import numpy as np

from vehicle_estimation.models.linear_single_track import LinearSingleTrackParameters
from .base import MeasurementModel


class LateralAccelerationMeasurement(MeasurementModel):
    source = "sensor.lateral_acceleration"

    def __init__(self, p: LinearSingleTrackParameters, sigma_mps2: float):
        self.p = p
        self.sigma = float(sigma_mps2)

    def _h_and_d(self, vx: float) -> tuple[np.ndarray, float]:
        p = self.p
        vx = max(float(vx), p.min_vx_mps)
        cf, cr, m, lf, lr = (
            p.cornering_stiffness_front_nprad,
            p.cornering_stiffness_rear_nprad,
            p.mass_kg,
            p.lf_m,
            p.lr_m,
        )
        H = np.array(
            [[-(cf + cr) / m, -(cf * lf - cr * lr) / (m * vx)]],
            dtype=float,
        )
        D = cf / m
        return H, D

    def predict(self, x: np.ndarray, *, delta: float, vx: float) -> np.ndarray:
        H, D = self._h_and_d(vx)
        return H @ np.asarray(x, dtype=float) + np.array([D * float(delta)])

    def jacobian_x(self, x: np.ndarray, *, delta: float, vx: float) -> np.ndarray:
        H, _ = self._h_and_d(vx)
        return H

    def default_covariance(self) -> np.ndarray:
        return np.array([[self.sigma**2]])
