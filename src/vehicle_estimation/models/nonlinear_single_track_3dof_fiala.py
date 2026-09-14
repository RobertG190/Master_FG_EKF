from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NonlinearSingleTrack3DOFParameters:
    """Parameters of the nonlinear planar 3-DoF single-track model."""

    mass_kg: float
    yaw_inertia_kgm2: float
    lf_m: float
    lr_m: float
    cornering_stiffness_front_nprad: float
    cornering_stiffness_rear_nprad: float
    road_friction_coefficient: float = 1.1
    gravity_mps2: float = 9.81
    min_vx_mps: float = 1.0


class NonlinearSingleTrack3DOF:
    """Nonlinear planar 3-DoF single-track model with Fiala lateral tires.

    State
    -----
    x = [vx, vy, r]

    Inputs
    ------
    delta : front road-wheel steering angle [rad]
    ax_rear_axle_mps2 : measured longitudinal acceleration at rear axle [m/s^2]

    The measured rear-axle longitudinal acceleration is converted to CoG
    acceleration using rigid-body kinematics:

        ax_CoG = ax_RA - lr * r^2

    Together with

        ax_CoG = vx_dot - r * vy

    this gives

        vx_dot = ax_RA - lr*r^2 + r*vy.

    Lateral tire forces are calculated with a Fiala-type nonlinear tire law
    using static front/rear axle normal loads.
    """

    state_names = (
        "longitudinal_velocity_mps",
        "lateral_velocity_mps",
        "yaw_rate_radps",
    )

    input_names = (
        "steering_angle_rad",
        "longitudinal_acceleration_rear_axle_mps2",
    )

    def __init__(self, p: NonlinearSingleTrack3DOFParameters):
        self.p = p

    @property
    def nx(self) -> int:
        return 3

    @staticmethod
    def _state(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape != (3,):
            raise ValueError(f"Expected state shape (3,), got {x.shape}")
        return x

    def _safe_vx(self, vx: float) -> float:
        vx = float(vx)

        if abs(vx) >= self.p.min_vx_mps:
            return vx

        return float(
            np.copysign(
                self.p.min_vx_mps,
                vx if vx != 0.0 else 1.0,
            )
        )

    def sideslip_angle(self, x: np.ndarray) -> float:
        """Vehicle sideslip angle beta at the CoG."""
        vx, vy, _ = self._state(x)
        return float(np.arctan2(vy, vx))

    def slip_angles(
        self,
        x: np.ndarray,
        *,
        delta: float,
    ) -> tuple[float, float]:
        """Front/rear tire slip angles using exact atan2 geometry."""

        vx, vy, r = self._state(x)
        vx_safe = self._safe_vx(vx)

        alpha_f = float(delta) - np.arctan2(
            vy + self.p.lf_m * r,
            vx_safe,
        )

        alpha_r = -np.arctan2(
            vy - self.p.lr_m * r,
            vx_safe,
        )

        return float(alpha_f), float(alpha_r)

    def static_axle_normal_loads(self) -> tuple[float, float]:
        """Static front/rear axle normal loads."""

        wheelbase = self.p.lf_m + self.p.lr_m

        fz_f = (
            self.p.mass_kg
            * self.p.gravity_mps2
            * self.p.lr_m
            / wheelbase
        )

        fz_r = (
            self.p.mass_kg
            * self.p.gravity_mps2
            * self.p.lf_m
            / wheelbase
        )

        return float(fz_f), float(fz_r)

    @staticmethod
    def tire_force(
        alpha_rad: float,
        cornering_stiffness_nprad: float,
        mu: float,
        normal_load_n: float,
    ) -> float:
        """Fiala-type nonlinear lateral tire force."""

        alpha = float(alpha_rad)
        C = float(cornering_stiffness_nprad)
        Fz = float(normal_load_n)
        mu = float(mu)

        if C <= 0.0:
            raise ValueError("cornering_stiffness_nprad must be positive")

        if Fz <= 0.0:
            raise ValueError("normal_load_n must be positive")

        if mu <= 0.0:
            raise ValueError("mu must be positive")

        tan_alpha = np.tan(alpha)

        # Slip angle at which the tire reaches full lateral-force saturation.
        alpha_sl = np.arctan(3.0 * mu * Fz / C)

        if abs(alpha) < alpha_sl:
            Fy = (
                C * tan_alpha
                - (C**2 / (3.0 * mu * Fz))
                * abs(tan_alpha)
                * tan_alpha
                + (C**3 / (27.0 * mu**2 * Fz**2))
                * tan_alpha**3
            )
        else:
            Fy = mu * Fz * np.sign(alpha)

        return float(Fy)

    def lateral_tire_forces(
        self,
        x: np.ndarray,
        *,
        delta: float,
    ) -> tuple[float, float]:
        """Return front/rear lateral tire forces."""

        alpha_f, alpha_r = self.slip_angles(
            x,
            delta=delta,
        )

        fz_f, fz_r = self.static_axle_normal_loads()

        fy_f = self.tire_force(
            alpha_f,
            self.p.cornering_stiffness_front_nprad,
            self.p.road_friction_coefficient,
            fz_f,
        )

        fy_r = self.tire_force(
            alpha_r,
            self.p.cornering_stiffness_rear_nprad,
            self.p.road_friction_coefficient,
            fz_r,
        )

        return float(fy_f), float(fy_r)

    def body_lateral_force_and_yaw_acceleration(
        self,
        x: np.ndarray,
        *,
        delta: float,
    ) -> tuple[float, float]:
        """Return total body lateral force and yaw acceleration r_dot."""

        fy_f, fy_r = self.lateral_tire_forces(
            x,
            delta=delta,
        )

        # Front lateral tire force must be rotated into the body frame.
        fy_f_body = fy_f * np.cos(float(delta))

        fy_total_body = fy_f_body + fy_r

        r_dot = (
            self.p.lf_m * fy_f_body
            - self.p.lr_m * fy_r
        ) / self.p.yaw_inertia_kgm2

        return float(fy_total_body), float(r_dot)

    def longitudinal_acceleration_cog(
        self,
        x: np.ndarray,
        *,
        ax_rear_axle_mps2: float,
    ) -> float:
        """Transform measured rear-axle longitudinal acceleration to the CoG."""

        _, _, r = self._state(x)

        return float(
            float(ax_rear_axle_mps2)
            - self.p.lr_m * r**2
        )

    def lateral_acceleration_cog(
        self,
        x: np.ndarray,
        *,
        delta: float,
    ) -> float:
        """Predicted lateral acceleration at the CoG."""

        fy_total, _ = self.body_lateral_force_and_yaw_acceleration(
            x,
            delta=delta,
        )

        return float(
            fy_total / self.p.mass_kg
        )

    def lateral_acceleration_rear_axle(
        self,
        x: np.ndarray,
        *,
        delta: float,
    ) -> float:
        """Predicted lateral acceleration at rear-axle center.

        With rear-axle position relative to the CoG

            r_RA/CoG = [-lr, 0, 0]

        the lateral acceleration is

            ay_RA = ay_CoG - lr * r_dot.
        """

        fy_total, r_dot = self.body_lateral_force_and_yaw_acceleration(
            x,
            delta=delta,
        )

        ay_cog = fy_total / self.p.mass_kg

        return float(
            ay_cog
            - self.p.lr_m * r_dot
        )

    def continuous_dynamics(
        self,
        x: np.ndarray,
        *,
        delta: float,
        ax_rear_axle_mps2: float,
    ) -> np.ndarray:
        """Continuous dynamics x_dot = f(x,u) for x=[vx, vy, r]."""

        vx, vy, r = self._state(x)

        fy_total, r_dot = self.body_lateral_force_and_yaw_acceleration(
            x,
            delta=delta,
        )

        # ax_CoG = vx_dot - r*vy
        # ax_CoG = ax_RA - lr*r^2
        ax_cog = (
            float(ax_rear_axle_mps2)
            - self.p.lr_m * r**2
        )

        vx_dot = ax_cog + r * vy

        # ay_CoG = vy_dot + r*vx = sum(Fy)/m
        vy_dot = (
            -r * vx
            + fy_total / self.p.mass_kg
        )

        return np.array(
            [vx_dot, vy_dot, r_dot],
            dtype=float,
        )

    def discrete_dynamics(
        self,
        x: np.ndarray,
        *,
        delta: float,
        ax_rear_axle_mps2: float,
        dt: float,
    ) -> np.ndarray:
        """Propagate one interval with fourth-order Runge-Kutta (RK4)."""

        if dt < 0.0:
            raise ValueError("dt must be nonnegative")

        x = self._state(x)

        if dt == 0.0:
            return x.copy()

        kwargs = {
            "delta": float(delta),
            "ax_rear_axle_mps2": float(ax_rear_axle_mps2),
        }

        k1 = self.continuous_dynamics(
            x,
            **kwargs,
        )

        k2 = self.continuous_dynamics(
            x + 0.5 * dt * k1,
            **kwargs,
        )

        k3 = self.continuous_dynamics(
            x + 0.5 * dt * k2,
            **kwargs,
        )

        k4 = self.continuous_dynamics(
            x + dt * k3,
            **kwargs,
        )

        return x + (
            dt / 6.0
        ) * (
            k1
            + 2.0 * k2
            + 2.0 * k3
            + k4
        )

    def discrete_jacobian_x(
        self,
        x: np.ndarray,
        *,
        delta: float,
        ax_rear_axle_mps2: float,
        dt: float,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Numerical EKF transition Jacobian d f_d / d x."""

        x = self._state(x)

        F = np.zeros(
            (3, 3),
            dtype=float,
        )

        for i in range(3):
            step = eps * max(
                1.0,
                abs(x[i]),
            )

            xp = x.copy()
            xm = x.copy()

            xp[i] += step
            xm[i] -= step

            fp = self.discrete_dynamics(
                xp,
                delta=delta,
                ax_rear_axle_mps2=ax_rear_axle_mps2,
                dt=dt,
            )

            fm = self.discrete_dynamics(
                xm,
                delta=delta,
                ax_rear_axle_mps2=ax_rear_axle_mps2,
                dt=dt,
            )

            F[:, i] = (
                fp - fm
            ) / (
                2.0 * step
            )

        return F
