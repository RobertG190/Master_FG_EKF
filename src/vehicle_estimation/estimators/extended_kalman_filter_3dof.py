from __future__ import annotations

import numpy as np
import pandas as pd

from vehicle_estimation.core.types import Estimate, Event, EventKind
from vehicle_estimation.models.nonlinear_single_track_3dof import NonlinearSingleTrack3DOF
from .base import Estimator


class VehicleExtendedKalmanFilter3DOF(Estimator):
    """Event-driven EKF for the nonlinear 3-DoF single-track model.

    State: x=[vx, vy, r]
    Inputs held piecewise constant between events:
      - input.steering_angle
      - input.longitudinal_acceleration_rear_axle
    Measurements:
      - sensor.longitudinal_velocity
      - sensor.yaw_rate
      - sensor.lateral_acceleration_rear_axle
    """

    INPUT_STEERING = "input.steering_angle"
    INPUT_AX_REAR = "input.longitudinal_acceleration_rear_axle"

    def __init__(
        self,
        model: NonlinearSingleTrack3DOF,
        sensors: dict[str, object],
        *,
        x0: np.ndarray,
        P0: np.ndarray,
        q_std: np.ndarray,
        t0: float = 0.0,
        initial_delta: float = 0.0,
        initial_ax_rear: float = 0.0,
    ):
        self.model = model
        self.sensors = sensors
        self.x = np.asarray(x0, dtype=float).reshape(3).copy()
        self.P = np.asarray(P0, dtype=float).reshape(3, 3).copy()
        self.q_std = np.asarray(q_std, dtype=float).reshape(3).copy()
        self.t = float(t0)
        self.delta = float(initial_delta)
        self.ax_rear = float(initial_ax_rear)
        self._history: list[Estimate] = []
        self._diag = {
            "prediction_steps": 0,
            "measurement_updates": 0,
            "rejected_oosm": 0,
            "unknown_events": 0,
        }

    def _predict_to(self, target_time: float) -> None:
        dt = float(target_time) - self.t
        if dt < -1e-12:
            self._diag["rejected_oosm"] += 1
            raise ValueError(
                f"Out-of-sequence event: measurement_time={target_time:.6f} "
                f"< filter_time={self.t:.6f}"
            )
        if dt <= 1e-12:
            return

        # EKF linearizes the nonlinear discrete transition around the current state.
        F = self.model.discrete_jacobian_x(
            self.x,
            delta=self.delta,
            ax_rear_axle_mps2=self.ax_rear,
            dt=dt,
        )
        self.x = self.model.discrete_dynamics(
            self.x,
            delta=self.delta,
            ax_rear_axle_mps2=self.ax_rear,
            dt=dt,
        )

        Qd = np.diag(self.q_std**2) * dt
        self.P = F @ self.P @ F.T + Qd
        self.P = 0.5 * (self.P + self.P.T)
        self.t = float(target_time)
        self._diag["prediction_steps"] += 1

    def _measurement_update(self, event: Event) -> None:
        sensor = self.sensors[event.source]
        z = np.asarray(event.value, dtype=float).reshape(-1)
        zhat = sensor.predict(self.x, delta=self.delta).reshape(-1)
        H = sensor.jacobian_x(self.x, delta=self.delta)
        R = (
            np.asarray(event.covariance, dtype=float)
            if event.covariance is not None
            else sensor.default_covariance()
        )

        innovation = z - zhat
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ innovation

        # Joseph covariance update for numerical robustness.
        I = np.eye(3)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self._diag["measurement_updates"] += 1
        self._history.append(Estimate(self.t, self.x.copy(), self.P.copy(), event.source))

    def process(self, event: Event) -> None:
        self._predict_to(event.measurement_time)

        if event.kind == EventKind.INPUT:
            value = float(np.asarray(event.value).reshape(-1)[0])
            if event.source == self.INPUT_STEERING:
                self.delta = value
            elif event.source == self.INPUT_AX_REAR:
                self.ax_rear = value
            else:
                self._diag["unknown_events"] += 1
            return

        if event.kind == EventKind.MEASUREMENT:
            if event.source not in self.sensors:
                self._diag["unknown_events"] += 1
                return
            self._measurement_update(event)
            return

        self._diag["unknown_events"] += 1

    @staticmethod
    def _beta_variance(x: np.ndarray, P: np.ndarray) -> float:
        vx, vy, _ = x
        den = vx * vx + vy * vy
        if den < 1e-12:
            return float("nan")
        J = np.array([[-vy / den, vx / den, 0.0]])
        return float((J @ P @ J.T)[0, 0])

    def estimates_frame(self) -> pd.DataFrame:
        rows = []
        for e in self._history:
            vx, vy, r = e.state
            beta = float(np.arctan2(vy, vx))
            rows.append(
                {
                    "time": e.time,
                    "trigger": e.trigger,
                    "longitudinal_velocity_mps": vx,
                    "lateral_velocity_mps": vy,
                    "yaw_rate_radps": r,
                    "beta_rad": beta,
                    "var_longitudinal_velocity": e.covariance[0, 0],
                    "var_lateral_velocity": e.covariance[1, 1],
                    "var_yaw_rate": e.covariance[2, 2],
                    "var_beta": self._beta_variance(e.state, e.covariance),
                }
            )
        return pd.DataFrame(rows)

    def diagnostics(self) -> dict:
        return dict(self._diag)
