from __future__ import annotations

from typing import Any

import gtsam
import numpy as np

from vehicle_estimation.sensors.parameterized_2dof import (
    LateralAccelerationMeasurement2DOF,
    YawRateMeasurement2DOF,
)


CustomFactor = getattr(gtsam, "CustomFactor")


def make_yaw_rate_factor(
    *,
    key_x: int,
    sensor: YawRateMeasurement2DOF,
    measurement: float,
    sigma: float,
):
    """
    Yaw-rate measurement factor.

    Variable:
        x = [beta, r]

    Measurement model:
        z = h(x) + v
        h(x) = r

    Residual:
        e = h(x) - z
    """

    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    noise_model = gtsam.noiseModel.Isotropic.Sigma(
        1,
        float(sigma),
    )

    def error_function(
        this: Any,
        values: gtsam.Values,
        jacobians: Any,
    ) -> np.ndarray:

        x = np.asarray(
            values.atVector(key_x),
            dtype=float,
        ).reshape(2)

        prediction = sensor.predict(x)

        residual = np.array(
            [
                prediction - float(measurement)
            ],
            dtype=float,
        )

        if jacobians is not None:

            hx = sensor.jacobian_x(x)

            jacobians[0] = np.asfortranarray(
                hx,
                dtype=float,
            )

        return residual

    return CustomFactor(
        noise_model,
        [key_x],
        error_function,
    )


def make_lateral_acceleration_factor(
    *,
    key_x: int,
    key_theta: int,
    sensor: LateralAccelerationMeasurement2DOF,
    measurement: float,
    delta: float,
    vx: float,
    sigma: float,
):
    """
    Lateral-acceleration measurement factor.

    Variables:
        x     = [beta, r]
        theta = [Cf, Cr]

    Known quantities:
        delta
        vx

    Measurement model:
        z = h(x, theta, delta, vx) + v

    Residual:
        e = h(x, theta, delta, vx) - z
    """

    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    noise_model = gtsam.noiseModel.Isotropic.Sigma(
        1,
        float(sigma),
    )

    def error_function(
        this: Any,
        values: gtsam.Values,
        jacobians: Any,
    ) -> np.ndarray:

        x = np.asarray(
            values.atVector(key_x),
            dtype=float,
        ).reshape(2)

        theta = np.asarray(
            values.atVector(key_theta),
            dtype=float,
        ).reshape(2)

        prediction = sensor.predict(
            x,
            theta,
            delta=delta,
            vx=vx,
        )

        residual = np.array(
            [
                prediction - float(measurement)
            ],
            dtype=float,
        )

        if jacobians is not None:

            hx = sensor.jacobian_x(
                x,
                theta,
                delta=delta,
                vx=vx,
            )

            htheta = sensor.jacobian_theta(
                x,
                theta,
                delta=delta,
                vx=vx,
            )

            jacobians[0] = np.asfortranarray(
                hx,
                dtype=float,
            )

            jacobians[1] = np.asfortranarray(
                htheta,
                dtype=float,
            )

        return residual

    return CustomFactor(
        noise_model,
        [
            key_x,
            key_theta,
        ],
        error_function,
    )