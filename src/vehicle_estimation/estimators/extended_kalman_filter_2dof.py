from __future__ import annotations

import numpy as np
import pandas as pd

from vehicle_estimation.core.types import Estimate, Event, EventKind
from vehicle_estimation.models.linear_single_track_parameter_estimation import (
    JointParametersSingleTrack2DOF,
)
from .base import Estimator


class VehicleExtendedKalmanFilter2DOFJoint(Estimator):
    """
    Event-driven EKF for joint state-parameter estimation
    using the 2-DoF single-track model.

    State:
        x = [beta, r, Cf, Cr]

        beta : sideslip angle [rad]
        r    : yaw rate [rad/s]
        Cf   : front axle cornering stiffness [N/rad]
        Cr   : rear axle cornering stiffness [N/rad]

    Held inputs / scheduling variables:
        - input.steering_angle
        - input.longitudinal_velocity

    Typical measurements:
        - sensor.yaw_rate
        - sensor.lateral_acceleration

    The longitudinal velocity vx is measured and treated as a
    scheduling variable, not as a state.
    """

    INPUT_STEERING = "input.steering_angle"
    INPUT_VX = "input.longitudinal_velocity"

    def __init__(
        self,
        model: JointParametersSingleTrack2DOF,
        sensors: dict[str, object],
        *,
        x0: np.ndarray,
        P0: np.ndarray,
        q_std: np.ndarray,
        t0: float = 0.0,
        initial_delta: float = 0.0,
        initial_vx: float = 0.0,
    ):
        self.model = model
        self.sensors = sensors

        # x = [beta, r, Cf, Cr]
        self.x = np.asarray(x0, dtype=float).reshape(4).copy()

        # Uncertainty of all four states.
        self.P = np.asarray(P0, dtype=float).reshape(4, 4).copy()

        # Process-noise standard deviations for:
        # beta, r, Cf, Cr
        self.q_std = np.asarray(q_std, dtype=float).reshape(4).copy()

        self.t = float(t0)

        # Inputs are held constant between input events.
        self.delta = float(initial_delta)
        self.vx = float(initial_vx)

        self._history: list[Estimate] = []

        self._diag = {
            "prediction_steps": 0,
            "measurement_updates": 0,
            "rejected_oosm": 0,
            "unknown_events": 0,
        }

    def _predict_to(self, target_time: float) -> None:
        """
        EKF prediction from current filter time self.t
        to target_time.
        """

        dt = float(target_time) - self.t

        if dt < -1e-12:
            self._diag["rejected_oosm"] += 1
            raise ValueError(
                f"Out-of-sequence event: measurement_time={target_time:.6f} "
                f"< filter_time={self.t:.6f}"
            )

        if dt <= 1e-12:
            return

        # ------------------------------------------------------------
        # 1. Linearize nonlinear joint dynamics around current estimate
        #
        # F_k = df_d/dx | x_hat
        #
        # F is 4x4 because:
        # x = [beta, r, Cf, Cr]
        # ------------------------------------------------------------
        F = self.model.discrete_jacobian_x(
            self.x,
            delta=self.delta,
            vx=self.vx,
            dt=dt,
        )

        # ------------------------------------------------------------
        # 2. State prediction
        #
        # x^-_{k+1} = f_d(x^+_k, u_k)
        # ------------------------------------------------------------
        self.x = self.model.discrete_dynamics(
            self.x,
            delta=self.delta,
            vx=self.vx,
            dt=dt,
        )

        # ------------------------------------------------------------
        # 3. Process-noise covariance
        #
        # Qd = diag(sigma_q^2) * dt
        #
        # For Cf/Cr this realizes approximately a random walk.
        # ------------------------------------------------------------
        Qd = np.diag(self.q_std**2) * dt

        # ------------------------------------------------------------
        # 4. Covariance prediction
        #
        # P^- = F P^+ F^T + Q
        # ------------------------------------------------------------
        self.P = F @ self.P @ F.T + Qd

        # Enforce numerical symmetry.
        self.P = 0.5 * (self.P + self.P.T)

        self.t = float(target_time)
        self._diag["prediction_steps"] += 1

    def _measurement_update(self, event: Event) -> None:
        """
        EKF measurement correction.
        """

        sensor = self.sensors[event.source]

        # Actual measurement z.
        z = np.asarray(event.value, dtype=float).reshape(-1)

        # ------------------------------------------------------------
        # Expected measurement:
        #
        # z_hat = h(x^-)
        # ------------------------------------------------------------
        zhat = sensor.predict(
            self.x,
            delta=self.delta,
            vx=self.vx,
        ).reshape(-1)

        # ------------------------------------------------------------
        # Linearization of measurement model:
        #
        # H = dh/dx
        # ------------------------------------------------------------
        H = sensor.jacobian_x(
            self.x,
            delta=self.delta,
            vx=self.vx,
        )

        # Use event-specific covariance if available,
        # otherwise sensor default.
        R = (
            np.asarray(event.covariance, dtype=float)
            if event.covariance is not None
            else sensor.default_covariance()
        )

        # ------------------------------------------------------------
        # Innovation
        #
        # nu = z - h(x^-)
        # ------------------------------------------------------------
        innovation = z - zhat

        # ------------------------------------------------------------
        # Innovation covariance
        #
        # S = H P^- H^T + R
        # ------------------------------------------------------------
        S = H @ self.P @ H.T + R

        # ------------------------------------------------------------
        # Kalman gain
        #
        # K = P^- H^T S^-1
        #
        # solve() avoids explicitly forming S^-1.
        # ------------------------------------------------------------
        K = np.linalg.solve(
            S,
            H @ self.P,
        ).T

        # ------------------------------------------------------------
        # State correction
        #
        # x^+ = x^- + K nu
        # ------------------------------------------------------------
        self.x = self.x + K @ innovation

        # ------------------------------------------------------------
        # Joseph covariance update
        #
        # P^+ =
        # (I-KH) P^- (I-KH)^T + K R K^T
        # ------------------------------------------------------------
        I = np.eye(4)
        IKH = I - K @ H

        self.P = (
            IKH @ self.P @ IKH.T
            + K @ R @ K.T
        )

        self.P = 0.5 * (self.P + self.P.T)

        self._diag["measurement_updates"] += 1

        self._history.append(
            Estimate(
                self.t,
                self.x.copy(),
                self.P.copy(),
                event.source,
            )
        )

    def process(self, event: Event) -> None:
        """
        Process one event from the canonical event stream.
        """

        # First propagate filter to physical measurement time.
        self._predict_to(event.measurement_time)

        # ------------------------------------------------------------
        # INPUT EVENT
        # ------------------------------------------------------------
        if event.kind == EventKind.INPUT:
            value = float(
                np.asarray(event.value).reshape(-1)[0]
            )

            if event.source == self.INPUT_STEERING:
                self.delta = value

            elif event.source == self.INPUT_VX:
                self.vx = value

            else:
                self._diag["unknown_events"] += 1

            return

        # ------------------------------------------------------------
        # MEASUREMENT EVENT
        # ------------------------------------------------------------
        if event.kind == EventKind.MEASUREMENT:
            if event.source not in self.sensors:
                self._diag["unknown_events"] += 1
                return

            self._measurement_update(event)
            return

        self._diag["unknown_events"] += 1

    def estimates_frame(self) -> pd.DataFrame:
        """
        Convert stored EKF states into a dataframe for evaluation.
        """

        rows = []

        for e in self._history:
            beta, r, cf, cr = e.state

            rows.append(
                {
                    "time": e.time,
                    "trigger": e.trigger,

                    "beta_rad": beta,
                    "yaw_rate_radps": r,

                    "cornering_stiffness_front_nprad": cf,
                    "cornering_stiffness_rear_nprad": cr,

                    "var_beta": e.covariance[0, 0],
                    "var_yaw_rate": e.covariance[1, 1],

                    "var_cornering_stiffness_front": (
                        e.covariance[2, 2]
                    ),
                    "var_cornering_stiffness_rear": (
                        e.covariance[3, 3]
                    ),
                }
            )

        return pd.DataFrame(rows)

    def diagnostics(self) -> dict:
        return dict(self._diag)