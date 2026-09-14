from __future__ import annotations

import numpy as np

from vehicle_estimation.models.parameterized_single_track_2dof import (
    ParameterizedSingleTrack2DOFParameters,
)
from vehicle_estimation.sensors.base import MeasurementModel


class YawRateMeasurement2DOF(MeasurementModel):
    """
    Yaw-rate measurement model.

    State:
        x = [beta, r]

    Measurement:
        z_r = r + v

    Predicted measurement:
        h(x) = r
    """

    def predict(
        self,
        x: np.ndarray,
    ) -> float:
        x = np.asarray(x, dtype=float).reshape(2)

        return float(x[1])

    def jacobian_x(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        _ = np.asarray(x, dtype=float).reshape(2)

        return np.array(
            [[0.0, 1.0]],
            dtype=float,
        )


class LateralAccelerationMeasurement2DOF(MeasurementModel):
    """
    CoG lateral-acceleration measurement model.

    State:
        x = [beta, r]

    Estimated parameters:
        theta = [Cf, Cr]

    Known quantities:
        delta : steering angle
        vx    : longitudinal velocity

    Measurement model:

        ay =
            -(Cf + Cr)/m * beta
            -(Cf*lf - Cr*lr)/(m*vx) * r
            + Cf/m * delta
    """

    def __init__(
        self,
        p: ParameterizedSingleTrack2DOFParameters,
    ):
        self.p = p

    def predict(
        self,
        x: np.ndarray,
        theta: np.ndarray,
        *,
        delta: float,
        vx: float,
    ) -> float:

        x = np.asarray(x, dtype=float).reshape(2)
        theta = np.asarray(theta, dtype=float).reshape(2)

        beta, r = x
        cf, cr = theta

        p = self.p

        vx = max(
            float(vx),
            p.min_vx_mps,
        )

        ay = (
            -(cf + cr) / p.mass_kg * beta
            - (
                cf * p.lf_m
                - cr * p.lr_m
            )
            / (p.mass_kg * vx)
            * r
            + cf / p.mass_kg * float(delta)
        )

        return float(ay)

    def jacobian_x(
        self,
        x: np.ndarray,
        theta: np.ndarray,
        *,
        delta: float,
        vx: float,
    ) -> np.ndarray:

        _ = np.asarray(x, dtype=float).reshape(2)
        theta = np.asarray(theta, dtype=float).reshape(2)

        cf, cr = theta

        p = self.p

        vx = max(
            float(vx),
            p.min_vx_mps,
        )

        d_ay_d_beta = (
            -(cf + cr)
            / p.mass_kg
        )

        d_ay_d_r = (
            -(
                cf * p.lf_m
                - cr * p.lr_m
            )
            / (p.mass_kg * vx)
        )

        return np.array(
            [[
                d_ay_d_beta,
                d_ay_d_r,
            ]],
            dtype=float,
        )

    def jacobian_theta(
        self,
        x: np.ndarray,
        theta: np.ndarray,
        *,
        delta: float,
        vx: float,
    ) -> np.ndarray:

        x = np.asarray(x, dtype=float).reshape(2)
        _ = np.asarray(theta, dtype=float).reshape(2)

        beta, r = x

        p = self.p

        vx = max(
            float(vx),
            p.min_vx_mps,
        )

        d_ay_d_cf = (
            float(delta)
            - beta
            - p.lf_m / vx * r
        ) / p.mass_kg

        d_ay_d_cr = (
            -beta
            + p.lr_m / vx * r
        ) / p.mass_kg

        return np.array(
            [[
                d_ay_d_cf,
                d_ay_d_cr,
            ]],
            dtype=float,
        )