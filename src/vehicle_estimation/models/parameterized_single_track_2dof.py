from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class ParameterizedSingleTrack2DOFParameters:
    """
    Known, fixed vehicle parameters.

    Cf and Cr are intentionally NOT included here.
    They are estimated parameters and are passed separately as theta.
    """

    mass_kg: float
    yaw_inertia_kgm2: float
    lf_m: float
    lr_m: float
    min_vx_mps: float = 1.0


class ParameterizedSingleTrack2DOF:
    """
    Parameterized linear single-track model.

    Dynamic state:
        x = [beta, r]

    Estimated physical parameters:
        theta = [Cf, Cr]

    Known input / scheduling variables:
        delta : steering angle
        vx    : longitudinal velocity

    Model:
        x[k+1] = f_d(x[k], theta, delta[k], vx[k], dt)
    """

    state_names = (
        "beta_rad",
        "yaw_rate_radps",
    )

    parameter_names = (
        "cornering_stiffness_front_nprad",
        "cornering_stiffness_rear_nprad",
    )

    def __init__(
        self,
        p: ParameterizedSingleTrack2DOFParameters,
    ):
        self.p = p

    def continuous_matrices(
        self,
        *,
        vx: float,
        theta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build continuous-time matrices

            x_dot = A(theta, vx) x + B(theta, vx) delta

        for

            x     = [beta, r]
            theta = [Cf, Cr]
        """

        theta = np.asarray(theta, dtype=float).reshape(2)

        cf = float(theta[0])
        cr = float(theta[1])

        p = self.p

        vx = max(
            float(vx),
            p.min_vx_mps,
        )

        m = p.mass_kg
        iz = p.yaw_inertia_kgm2
        lf = p.lf_m
        lr = p.lr_m

        a11 = -(cf + cr) / (m * vx)

        a12 = (
            -1.0
            - (cf * lf - cr * lr)
            / (m * vx**2)
        )

        a21 = (
            -(cf * lf - cr * lr)
            / iz
        )

        a22 = (
            -(cf * lf**2 + cr * lr**2)
            / (iz * vx)
        )

        b1 = cf / (m * vx)

        b2 = cf * lf / iz

        A = np.array(
            [
                [a11, a12],
                [a21, a22],
            ],
            dtype=float,
        )

        B = np.array(
            [
                [b1],
                [b2],
            ],
            dtype=float,
        )

        return A, B

    def continuous_dynamics(
        self,
        x: np.ndarray,
        theta: np.ndarray,
        *,
        delta: float,
        vx: float,
    ) -> np.ndarray:
        """
        Evaluate

            x_dot = f(x, theta, u)

        in continuous time.
        """

        x = np.asarray(x, dtype=float).reshape(2)

        A, B = self.continuous_matrices(
            vx=vx,
            theta=theta,
        )

        return (
            A @ x
            + B[:, 0] * float(delta)
        )

    def transition_matrices(
        self,
        *,
        vx: float,
        theta: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Exact zero-order-hold discretization:

            x[k+1] = Phi x[k] + Gamma delta[k]
        """

        if dt < 0.0:
            raise ValueError(
                "dt must be nonnegative"
            )

        if dt == 0.0:
            return (
                np.eye(2),
                np.zeros((2, 1)),
            )

        A, B = self.continuous_matrices(
            vx=vx,
            theta=theta,
        )

        # Augmented matrix for exact ZOH discretization:
        #
        #       [ A  B ]
        #   M = [      ]
        #       [ 0  0 ]
        #
        # exp(M dt) gives:
        #
        #       [ Phi  Gamma ]
        #       [   0      1 ]
        #

        M = np.zeros(
            (3, 3),
            dtype=float,
        )

        M[:2, :2] = A
        M[:2, 2:] = B

        Md = expm(M * float(dt))

        Phi = Md[:2, :2]
        Gamma = Md[:2, 2:]

        return Phi, Gamma

    def discrete_dynamics(
        self,
        x: np.ndarray,
        theta: np.ndarray,
        *,
        delta: float,
        vx: float,
        dt: float,
    ) -> np.ndarray:
        """
        Discrete vehicle transition:

            x[k+1] = f_d(x[k], theta, u[k])

        This is the function that the dynamics factor will use.
        """

        x = np.asarray(x, dtype=float).reshape(2)
        theta = np.asarray(theta, dtype=float).reshape(2)

        Phi, Gamma = self.transition_matrices(
            vx=vx,
            theta=theta,
            dt=dt,
        )

        x_next = (
            Phi @ x
            + Gamma[:, 0] * float(delta)
        )

        return x_next

    def discrete_jacobian_x(
        self,
        x: np.ndarray,
        theta: np.ndarray,
        *,
        delta: float,
        vx: float,
        dt: float,
    ) -> np.ndarray:
        """
        Jacobian of the discrete dynamics with respect to x:

            Fx = df_d / dx

        Since the model is linear in [beta, r] for fixed Cf and Cr,

            Fx = Phi
        exactly.
        """

        _ = np.asarray(x, dtype=float).reshape(2)

        Phi, _ = self.transition_matrices(
            vx=vx,
            theta=theta,
            dt=dt,
        )

        return Phi

    def discrete_jacobian_theta(
        self,
        x: np.ndarray,
        theta: np.ndarray,
        *,
        delta: float,
        vx: float,
        dt: float,
    ) -> np.ndarray:
        """
        Jacobian of the discrete dynamics with respect to

            theta = [Cf, Cr]

        Returns:

            J_theta = df_d / dtheta

        Shape:
            (2, 2)

        Numerical central differences are used initially because the exact
        discrete transition contains exp(A(theta) dt), whose analytic
        derivative is more cumbersome.
        """

        x = np.asarray(x, dtype=float).reshape(2)
        theta = np.asarray(theta, dtype=float).reshape(2)

        J = np.zeros(
            (2, 2),
            dtype=float,
        )

        for i in range(2):

            eps = max(
                1.0,
                abs(theta[i]) * 1e-6,
            )

            theta_plus = theta.copy()
            theta_minus = theta.copy()

            theta_plus[i] += eps
            theta_minus[i] -= eps

            f_plus = self.discrete_dynamics(
                x,
                theta_plus,
                delta=delta,
                vx=vx,
                dt=dt,
            )

            f_minus = self.discrete_dynamics(
                x,
                theta_minus,
                delta=delta,
                vx=vx,
                dt=dt,
            )

            J[:, i] = (
                f_plus - f_minus
            ) / (2.0 * eps)

        return J