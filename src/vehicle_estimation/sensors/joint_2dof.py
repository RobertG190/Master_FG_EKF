from __future__ import annotations

import numpy as np

from vehicle_estimation.models.linear_single_track_parameter_estimation import (
    JointParametersSingleTrack2DOF,
)
from .base import MeasurementModel


class JointYawRateMeasurement(MeasurementModel):
    """
    Yaw-rate measurement model for

        x = [beta, r, Cf, Cr]

    Measurement:
        z_r = r + v
    """

    source = "sensor.yaw_rate"

    def __init__(self, sigma_radps: float):
        self.sigma = float(sigma_radps)

    def predict(
        self,
        x: np.ndarray,
        *,
        delta: float,
        vx: float,
    ) -> np.ndarray:
        """
        Measurement function:

            h_r(x) = r
        """

        x = np.asarray(x, dtype=float).reshape(4)

        r = x[1]

        return np.array([r])

    def jacobian_x(
        self,
        x: np.ndarray,
        *,
        delta: float,
        vx: float,
    ) -> np.ndarray:
        """
        Measurement Jacobian:

            H_r = dh_r/dx
                = [0, 1, 0, 0]
        """

        return np.array(
            [[0.0, 1.0, 0.0, 0.0]],
            dtype=float,
        )

    def default_covariance(self) -> np.ndarray:
        """
        Measurement-noise covariance:

            R = sigma_r^2
        """

        return np.array(
            [[self.sigma**2]],
            dtype=float,
        )


class JointLateralAccelerationMeasurement(MeasurementModel):
    """
    CoG lateral-acceleration measurement model for

        x = [beta, r, Cf, Cr]

    Based on the linear 2-DoF single-track model.

    Measurement:

        ay =
            -(Cf + Cr)/m * beta
            -(Cf*lf - Cr*lr)/(m*vx) * r
            + Cf/m * delta

    Note:
        This is the CoG/vehicle lateral-acceleration model corresponding to
        source = "sensor.lateral_acceleration".

        It is NOT the rear-axle lateral-acceleration model.
    """

    source = "sensor.lateral_acceleration"

    def __init__(
        self,
        model: JointParametersSingleTrack2DOF,
        sigma_mps2: float,
    ):
        self.model = model
        self.p = model.p
        self.sigma = float(sigma_mps2)

    def predict(
        self,
        x: np.ndarray,
        *,
        delta: float,
        vx: float,
    ) -> np.ndarray:
        """
        Nonlinear measurement function:

            z_hat = h(x, delta, vx)
        """

        x = np.asarray(x, dtype=float).reshape(4)

        beta = float(x[0])
        r = float(x[1])
        cf = float(x[2])
        cr = float(x[3])

        p = self.p

        m = p.mass_kg
        lf = p.lf_m
        lr = p.lr_m

        vx = max(float(vx), p.min_vx_mps)
        delta = float(delta)

        ay = (
            -(cf + cr) / m * beta
            - (cf * lf - cr * lr) / (m * vx) * r
            + cf / m * delta
        )

        return np.array([ay], dtype=float)

    def jacobian_x(
        self,
        x: np.ndarray,
        *,
        delta: float,
        vx: float,
    ) -> np.ndarray:
        """
        Measurement Jacobian:

            H = dh/dx

        with:

            x = [beta, r, Cf, Cr]
        """

        x = np.asarray(x, dtype=float).reshape(4)

        beta = float(x[0])
        r = float(x[1])
        cf = float(x[2])
        cr = float(x[3])

        p = self.p

        m = p.mass_kg
        lf = p.lf_m
        lr = p.lr_m

        vx = max(float(vx), p.min_vx_mps)
        delta = float(delta)

        # d ay / d beta
        dh_dbeta = -(cf + cr) / m

        # d ay / d r
        dh_dr = (
            -(cf * lf - cr * lr)
            / (m * vx)
        )

        # d ay / d Cf
        dh_dcf = (
            delta
            - beta
            - lf * r / vx
        ) / m

        # d ay / d Cr
        dh_dcr = (
            -beta
            + lr * r / vx
        ) / m

        H = np.array(
            [[
                dh_dbeta,
                dh_dr,
                dh_dcf,
                dh_dcr,
            ]],
            dtype=float,
        )

        return H

    def default_covariance(self) -> np.ndarray:
        """
        Measurement-noise covariance:

            R = sigma_ay^2
        """

        return np.array(
            [[self.sigma**2]],
            dtype=float,
        )