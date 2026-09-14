from __future__ import annotations

from typing import Any

import gtsam
import numpy as np

from vehicle_estimation.models.parameterized_single_track_2dof import (
    ParameterizedSingleTrack2DOF,
)

# Some GTSAM Python type stubs do not expose CustomFactor even though
# the compiled module provides it at runtime.
CustomFactor = getattr(gtsam, "CustomFactor")


def make_single_track_dynamics_factor(
    *,
    key_xk: int,
    key_xk1: int,
    key_theta: int,
    model: ParameterizedSingleTrack2DOF,
    delta: float,
    vx: float,
    dt: float,
    sigma_beta: float,
    sigma_yaw_rate: float,
) -> gtsam.NonlinearFactor:
    """GTSAM dynamics factor for the parameterized 2-DoF single-track model.

    Variables:
        x[k]   = [beta_k, r_k]
        x[k+1] = [beta_k+1, r_k+1]
        theta  = [Cf, Cr]

    Residual:
        e = x[k+1] - f_d(x[k], theta, u[k])
    """

    if sigma_beta <= 0.0:
        raise ValueError("sigma_beta must be positive")
    if sigma_yaw_rate <= 0.0:
        raise ValueError("sigma_yaw_rate must be positive")
    if dt < 0.0:
        raise ValueError("dt must be nonnegative")

    noise_model = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([sigma_beta, sigma_yaw_rate], dtype=float)
    )

    def error_function(
        this: Any,
        values: gtsam.Values,
        jacobians: Any,
    ) -> np.ndarray:
        xk = np.asarray(
            values.atVector(key_xk),
            dtype=float,
        ).reshape(2)

        xk1 = np.asarray(
            values.atVector(key_xk1),
            dtype=float,
        ).reshape(2)

        theta = np.asarray(
            values.atVector(key_theta),
            dtype=float,
        ).reshape(2)

        x_pred = model.discrete_dynamics(
            xk,
            theta,
            delta=delta,
            vx=vx,
            dt=dt,
        )

        residual = xk1 - x_pred

        if jacobians is not None:
            fx = model.discrete_jacobian_x(
                xk,
                theta,
                delta=delta,
                vx=vx,
                dt=dt,
            )

            ftheta = model.discrete_jacobian_theta(
                xk,
                theta,
                delta=delta,
                vx=vx,
                dt=dt,
            )

            # e = x_{k+1} - f_d(x_k, theta)
            # de/dx_k   = -df_d/dx_k
            # de/dx_k+1 = I
            # de/dtheta = -df_d/dtheta
            jacobians[0] = np.asfortranarray(-fx, dtype=float)
            jacobians[1] = np.asfortranarray(np.eye(2), dtype=float)
            jacobians[2] = np.asfortranarray(-ftheta, dtype=float)

        return np.asarray(residual, dtype=float)

    return CustomFactor(
        noise_model,
        [key_xk, key_xk1, key_theta],
        error_function,
    )