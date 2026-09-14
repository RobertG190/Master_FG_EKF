from __future__ import annotations

import numpy as np
import pandas as pd

from vehicle_estimation.core.types import Estimate, Event, EventKind
from vehicle_estimation.models.base import DynamicModel
from vehicle_estimation.sensors.base import MeasurementModel
from .base import Estimator


class LinearVehicleKalmanFilter(Estimator):
    """Event-driven KF for the LPV 2-DoF single-track model.

    Current version assumes events are processed in arrival order and rejects
    out-of-sequence measurement timestamps. This is intentional: an OOSM-aware
    baseline can later be added as a separate estimator without changing the
    event/data interfaces.
    """

    INPUT_STEERING = "input.steering_angle"
    INPUT_VX = "input.longitudinal_velocity"

    def __init__(
        self,
        model: DynamicModel,
        sensors: dict[str, MeasurementModel],
        *,
        x0: np.ndarray,
        P0: np.ndarray,
        q_std: np.ndarray,
        t0: float = 0.0,
        initial_delta: float = 0.0,
        initial_vx: float = 10.0,
    ):
        self.model = model
        self.sensors = sensors
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        self.q_std = np.asarray(q_std, dtype=float).copy()
        self.t = float(t0)
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
        dt = float(target_time) - self.t
        if dt < -1e-12:
            self._diag["rejected_oosm"] += 1
            raise ValueError(
                f"Out-of-sequence event: measurement_time={target_time:.6f} < filter_time={self.t:.6f}"
            )
        if dt <= 1e-12:
            return

        F, B = self.model.transition_matrices(delta=self.delta, vx=self.vx, dt=dt)
        self.x = F @ self.x + B[:, 0] * self.delta

        # Continuous-time white process noise intensity, approximated over dt.
        Qd = np.diag(self.q_std**2) * dt
        self.P = F @ self.P @ F.T + Qd
        self.P = 0.5 * (self.P + self.P.T)
        self.t = float(target_time)
        self._diag["prediction_steps"] += 1

    def _measurement_update(self, event: Event) -> None:
        sensor = self.sensors[event.source]
        z = np.asarray(event.value, dtype=float).reshape(-1)
        zhat = sensor.predict(self.x, delta=self.delta, vx=self.vx).reshape(-1)
        H = sensor.jacobian_x(self.x, delta=self.delta, vx=self.vx)
        R = (
            np.asarray(event.covariance, dtype=float)
            if event.covariance is not None
            else sensor.default_covariance()
        )

        innovation = z - zhat
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ innovation

        # Joseph form preserves symmetry/PSD better numerically.
        I = np.eye(self.P.shape[0])
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self._diag["measurement_updates"] += 1
        self._history.append(
            Estimate(self.t, self.x.copy(), self.P.copy(), trigger=event.source)
        )

    def process(self, event: Event) -> None:
        self._predict_to(event.measurement_time)

        if event.kind == EventKind.INPUT:
            value = float(np.asarray(event.value).reshape(-1)[0])
            if event.source == self.INPUT_STEERING:
                self.delta = value
            elif event.source == self.INPUT_VX:
                self.vx = value
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

    def estimates_frame(self) -> pd.DataFrame:
        rows = []
        for e in self._history:
            rows.append(
                {
                    "time": e.time,
                    "trigger": e.trigger,
                    "beta_rad": e.state[0],
                    "yaw_rate_radps": e.state[1],
                    "var_beta": e.covariance[0, 0],
                    "var_yaw_rate": e.covariance[1, 1],
                    "cov_beta_yaw_rate": e.covariance[0, 1],
                }
            )
        return pd.DataFrame(rows)

    def diagnostics(self) -> dict:
        return dict(self._diag)
