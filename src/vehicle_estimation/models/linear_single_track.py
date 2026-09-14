from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .base import DynamicModel


@dataclass(frozen=True)
class LinearSingleTrackParameters:
    mass_kg: float
    yaw_inertia_kgm2: float
    lf_m: float
    lr_m: float
    cornering_stiffness_front_nprad: float
    cornering_stiffness_rear_nprad: float
    min_vx_mps: float = 1.0


class LinearSingleTrack2DOF(DynamicModel):
    """Linear/LPV bicycle model with x = [beta, r].

    Sign convention:
    - positive steering delta produces positive lateral force / positive yaw rate,
    - linear tire forces are based on positive effective slip convention used by
      the standard state-space form implemented below.

    vx is treated as a measured scheduling variable, not as a state.
    """

    state_names = ("beta_rad", "yaw_rate_radps")

    def __init__(self, p: LinearSingleTrackParameters):
        self.p = p

    def continuous_matrices(self, vx: float) -> tuple[np.ndarray, np.ndarray]:
        p = self.p
        vx = max(float(vx), p.min_vx_mps)
        m, iz, lf, lr, cf, cr = (
            p.mass_kg,
            p.yaw_inertia_kgm2,
            p.lf_m,
            p.lr_m,
            p.cornering_stiffness_front_nprad,
            p.cornering_stiffness_rear_nprad,
        )

        a11 = -(cf + cr) / (m * vx)
        a12 = -1.0 - (cf * lf - cr * lr) / (m * vx**2)
        a21 = -(cf * lf - cr * lr) / iz
        a22 = -(cf * lf**2 + cr * lr**2) / (iz * vx)

        b1 = cf / (m * vx)
        b2 = cf * lf / iz

        A = np.array([[a11, a12], [a21, a22]], dtype=float)
        B = np.array([[b1], [b2]], dtype=float)
        return A, B

    def transition_matrices(
        self, *, delta: float, vx: float, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if dt < 0:
            raise ValueError("dt must be nonnegative")
        if dt == 0:
            return np.eye(2), np.zeros((2, 1))

        A, B = self.continuous_matrices(vx)

        # Exact ZOH discretization through an augmented matrix exponential.
        M = np.zeros((3, 3), dtype=float)
        M[:2, :2] = A
        M[:2, 2:] = B
        Md = expm(M * dt)
        F = Md[:2, :2]
        Bd = Md[:2, 2:]
        return F, Bd

    def discrete_dynamics(
        self, x: np.ndarray, *, delta: float, vx: float, dt: float
    ) -> np.ndarray:
        F, B = self.transition_matrices(delta=delta, vx=vx, dt=dt)
        return F @ np.asarray(x, dtype=float) + B[:, 0] * float(delta)
