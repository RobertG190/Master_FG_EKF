from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .base import DynamicModel


@dataclass(frozen=True)
class JointSingleTrackParameters:
    mass_kg: float
    yaw_inertia_kgm2: float
    lf_m: float
    lr_m: float
    min_vx_mps: float = 1.0


class JointParametersSingleTrack2DOF(DynamicModel):
    """
    Joint state-parameter 2DOF bicycle model.

    State:
        x = [beta, r, Cf, Cr]

    beta : sideslip angle [rad]
    r    : yaw rate [rad/s]
    Cf   : front axle cornering stiffness [N/rad]
    Cr   : rear axle cornering stiffness [N/rad]

    Inputs:
        delta : steering angle [rad]
        vx    : measured longitudinal velocity [m/s]
    """

    state_names = ("beta_rad", "yaw_rate_radps", "cornering_stiffness_front_nprad","cornering_stiffness_rear_nprad")

    def __init__(self, p: JointSingleTrackParameters):
        self.p = p

    def continuous_matrices(self, vx: float, cf: float, cr: float) -> tuple[np.ndarray, np.ndarray]:
        p = self.p
        vx = max(float(vx), p.min_vx_mps)
        m, iz, lf, lr = (
            p.mass_kg,
            p.yaw_inertia_kgm2,
            p.lf_m,
            p.lr_m,
     
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
        self, *, vx: float,cf: float, cr:float, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if dt < 0:
            raise ValueError("dt must be nonnegative")
        if dt == 0:
            return np.eye(2), np.zeros((2, 1))

        A, B = self.continuous_matrices(vx=vx, cf=cf, cr=cr)

        # Exact ZOH discretization through an augmented matrix exponential.
        M = np.zeros((3, 3), dtype=float)
        M[:2, :2] = A
        M[:2, 2:] = B
        Md = expm(M * dt)
        F = Md[:2, :2]
        Bd = Md[:2, 2:]
        return F, Bd

    def discrete_dynamics(
            self,
            x: np.ndarray,
            *,
            delta: float,
            vx: float,
            dt: float,
        ) -> np.ndarray:

            x = np.asarray(x, dtype=float)

            beta = x[0]
            r = x[1]
            cf = x[2]
            cr = x[3]

            if cf <= 0.0 or cr <= 0.0:
                raise ValueError("Cornering stiffness must be positive")

            F, B = self.transition_matrices(
                vx=vx,
                cf=cf,
                cr=cr,
                dt=dt,
            )

            vehicle_state = np.array([beta, r])

            vehicle_next = (
                F @ vehicle_state
                + B[:, 0] * float(delta)
            )

            return np.array(
                [
                    vehicle_next[0],
                    vehicle_next[1],
                    cf,
                    cr,
                ]
            )
    def discrete_jacobian_x(
            self,
            x: np.ndarray,
            *,
            delta: float,
            vx: float,
            dt: float,
        ) -> np.ndarray:

            x = np.asarray(x, dtype=float)

            n = len(x)

            J = np.zeros((n, n))

            for i in range(n):

                if i < 2:
                    eps = max(1e-7, abs(x[i]) * 1e-6)
                else:
                    eps = max(1.0, abs(x[i]) * 1e-6)

                xp = x.copy()
                xm = x.copy()

                xp[i] += eps
                xm[i] -= eps

                fp = self.discrete_dynamics(
                    xp,
                    delta=delta,
                    vx=vx,
                    dt=dt,
                )

                fm = self.discrete_dynamics(
                    xm,
                    delta=delta,
                    vx=vx,
                    dt=dt,
                )

                J[:, i] = (fp - fm) / (2.0 * eps)

            return J