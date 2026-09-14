from __future__ import annotations

import numpy as np

from vehicle_estimation.models.linear_single_track import LinearSingleTrack2DOF
from .base import MeasurementModel


class RearAxleLateralAccelerationMeasurement(MeasurementModel):
    """Lateral acceleration measured at the rear-axle center.

    The dynamic single-track model state is x=[beta, r] at the CoG. For a
    rigid body, with the rear axle located x=-l_r from the CoG,

        a_y,RA = a_y,CoG - l_r * dot(r)

    (ISO 8855: x forward, y left, z up).

    Both a_y,CoG and dot(r) are linear in [beta, r, delta] for the selected
    2-DoF model, so this remains a linear measurement equation and no noisy
    numerical differentiation of yaw rate is required.
    """

    source = "sensor.lateral_acceleration_rear_axle"

    def __init__(self, model: LinearSingleTrack2DOF, sigma_mps2: float):
        self.model = model
        self.p = model.p
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
        h_cog = np.array(
            [[-(cf + cr) / m, -(cf * lf - cr * lr) / (m * vx)]],
            dtype=float,
        )
        d_cog = cf / m

        A, B = self.model.continuous_matrices(vx)
        h_ra = h_cog - lr * A[1:2, :]
        d_ra = d_cog - lr * float(B[1, 0])
        return h_ra, d_ra

    def predict(self, x: np.ndarray, *, delta: float, vx: float) -> np.ndarray:
        H, D = self._h_and_d(vx)
        return H @ np.asarray(x, dtype=float) + np.array([D * float(delta)])

    def jacobian_x(self, x: np.ndarray, *, delta: float, vx: float) -> np.ndarray:
        H, _ = self._h_and_d(vx)
        return H

    def default_covariance(self) -> np.ndarray:
        return np.array([[self.sigma**2]])
