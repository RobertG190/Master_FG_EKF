from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from vehicle_estimation.data.canonical_hdf5 import CanonicalHDF5Dataset
from vehicle_estimation.data.synthetic import Synthetic2DOFDataset

from vehicle_estimation.estimators.extended_kalman_filter_3dof import (
    VehicleExtendedKalmanFilter3DOF,
)
from vehicle_estimation.estimators.kalman_filter import LinearVehicleKalmanFilter
from vehicle_estimation.estimators.extended_kalman_filter_2dof import (
    VehicleExtendedKalmanFilter2DOFJoint,
)

from vehicle_estimation.models.linear_single_track import (
    LinearSingleTrack2DOF,
    LinearSingleTrackParameters,
)
from vehicle_estimation.models.linear_single_track_parameter_estimation import (
    JointParametersSingleTrack2DOF,
    JointSingleTrackParameters,
)

# 3-DoF with linear lateral tire law
from vehicle_estimation.models.nonlinear_single_track_3dof import (
    NonlinearSingleTrack3DOF as NonlinearSingleTrack3DOFLinearTire,
    NonlinearSingleTrack3DOFParameters as NonlinearSingleTrack3DOFLinearTireParameters,
)

# 3-DoF with nonlinear Fiala tire law
from vehicle_estimation.models.nonlinear_single_track_3dof_fiala import (
    NonlinearSingleTrack3DOF as NonlinearSingleTrack3DOFFiala,
    NonlinearSingleTrack3DOFParameters as NonlinearSingleTrack3DOFFialaParameters,
)

from vehicle_estimation.sensors.lateral_acceleration import (
    LateralAccelerationMeasurement,
)
from vehicle_estimation.sensors.nonlinear_3dof import (
    LongitudinalVelocityMeasurement3DOF,
    RearAxleLateralAccelerationMeasurement3DOF,
    YawRateMeasurement3DOF,
)
from vehicle_estimation.sensors.rear_axle_lateral_acceleration import (
    RearAxleLateralAccelerationMeasurement,
)
from vehicle_estimation.sensors.yaw_rate import YawRateMeasurement
from vehicle_estimation.sensors.joint_2dof import (
    JointLateralAccelerationMeasurement,
    JointYawRateMeasurement,
)


def _model_parameters(cfg: dict) -> dict:
    """Load model parameters either from parameter_file or inline config."""
    mcfg = cfg["model"]

    if "parameter_file" in mcfg:
        path = Path(mcfg["parameter_file"])

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"Parameter file must contain a mapping: {path}"
            )

        return data

    return mcfg["parameters"]


def _build_dataset(
    cfg: dict,
    *,
    model=None,
    yaw_sensor=None,
    ay_sensor=None,
):
    """Build the configured dataset adapter."""
    dcfg = cfg["dataset"]

    if dcfg["type"] == "canonical_hdf5":
        return CanonicalHDF5Dataset(
            dcfg["path"],
            enabled_sources=dcfg.get("enabled_sources"),
            stride=int(dcfg.get("stride", 1)),
            start_s=dcfg.get("start_s"),
            end_s=dcfg.get("end_s"),
        )

    if dcfg["type"] == "synthetic_2dof":
        if model is None or yaw_sensor is None or ay_sensor is None:
            raise ValueError(
                "synthetic_2dof requires the linear 2-DoF stack"
            )

        return Synthetic2DOFDataset(
            model,
            yaw_sensor,
            ay_sensor,
            duration_s=float(dcfg["duration_s"]),
            dt_s=float(dcfg["dt_s"]),
            seed=int(cfg["run"]["seed"]),
            vx_base_mps=float(dcfg["vx_base_mps"]),
            vx_variation_mps=float(
                dcfg.get("vx_variation_mps", 0.0)
            ),
            steering_amplitude_rad=float(
                dcfg["steering_amplitude_rad"]
            ),
            steering_frequency_hz=float(
                dcfg["steering_frequency_hz"]
            ),
            truth_process_std=np.array(
                dcfg.get("truth_process_std", [0.0, 0.0]),
                dtype=float,
            ),
        )

    raise ValueError(
        f"Unknown dataset type: {dcfg['type']}. "
        "Raw datasets should first be converted to canonical_hdf5."
    )


def _common_parameter_kwargs(mp: dict) -> dict:
    """Parameters shared by the current single-track models."""
    return dict(
        mass_kg=float(mp["mass_kg"]),
        yaw_inertia_kgm2=float(mp["yaw_inertia_kgm2"]),
        lf_m=float(mp["lf_m"]),
        lr_m=float(mp["lr_m"]),
        cornering_stiffness_front_nprad=float(
            mp["cornering_stiffness_front_nprad"]
        ),
        cornering_stiffness_rear_nprad=float(
            mp["cornering_stiffness_rear_nprad"]
        ),
        min_vx_mps=float(mp.get("min_vx_mps", 1.0)),
    )



def _joint_parameter_kwargs(mp: dict) -> dict:
    """Known vehicle parameters for the joint state-parameter 2-DoF model.

    Cf and Cr are intentionally NOT loaded here: they are EKF states and their
    initial guesses belong in the experiment configuration.
    """
    return dict(
        mass_kg=float(mp["mass_kg"]),
        yaw_inertia_kgm2=float(mp["yaw_inertia_kgm2"]),
        lf_m=float(mp["lf_m"]),
        lr_m=float(mp["lr_m"]),
        min_vx_mps=float(mp.get("min_vx_mps", 1.0)),
    )


def _build_3dof_ekf_stack(
    cfg: dict,
    model,
    model_type: str,
):
    """Build sensors, dataset and EKF shared by both 3-DoF models."""

    estimator_type = cfg["estimator"].get(
        "type",
        "extended_kalman_filter_3dof",
    )

    if estimator_type not in {
        "extended_kalman_filter_3dof",
        "ekf_3dof",
    }:
        raise ValueError(
            f"Estimator {estimator_type!r} "
            f"is not configured for {model_type!r}"
        )

    # --- Sensors ---------------------------------------------------------

    vx_sensor = LongitudinalVelocityMeasurement3DOF(
        float(
            cfg["sensors"]["longitudinal_velocity"]["sigma_mps"]
        )
    )

    yaw_sensor = YawRateMeasurement3DOF(
        float(
            cfg["sensors"]["yaw_rate"]["sigma_radps"]
        )
    )

    ay_cfg = cfg["sensors"]["lateral_acceleration"]

    if ay_cfg.get("location", "rear_axle") != "rear_axle":
        raise ValueError(
            "The current 3-DoF EKF expects rear-axle lateral acceleration"
        )

    ay_sensor = RearAxleLateralAccelerationMeasurement3DOF(
        model,
        float(ay_cfg["sigma_mps2"]),
    )

    sensors = {
        vx_sensor.source: vx_sensor,
        yaw_sensor.source: yaw_sensor,
        ay_sensor.source: ay_sensor,
    }

    # --- Dataset ---------------------------------------------------------

    dataset = _build_dataset(cfg)

    # --- EKF -------------------------------------------------------------

    ecfg = cfg["estimator"]

    estimator = VehicleExtendedKalmanFilter3DOF(
        model,
        sensors,
        x0=np.array(
            ecfg["initial_state"],
            dtype=float,
        ),
        P0=np.diag(
            np.array(
                ecfg["initial_std"],
                dtype=float,
            )
            ** 2
        ),
        q_std=np.array(
            ecfg["process_noise_std_per_sqrt_s"],
            dtype=float,
        ),
        t0=float(
            ecfg.get("initial_time_s", 0.0)
        ),
        initial_delta=float(
            ecfg.get(
                "initial_steering_angle_rad",
                0.0,
            )
        ),
        initial_ax_rear=float(
            ecfg.get(
                "initial_longitudinal_acceleration_rear_axle_mps2",
                0.0,
            )
        ),
    )

    return dataset, model, sensors, estimator


def build_stack(cfg: dict):
    """Build dataset, model, sensors and estimator from configuration."""

    mp = _model_parameters(cfg)

    model_type = cfg["model"].get(
        "type",
        "linear_single_track_2dof",
    )

    estimator_type = cfg["estimator"].get(
        "type",
        "kalman_filter",
    )

    # =====================================================================
    # 1) LINEAR 2-DoF SINGLE TRACK + STANDARD KF
    # =====================================================================

    if model_type == "linear_single_track_2dof":

        p = LinearSingleTrackParameters(
            **_common_parameter_kwargs(mp)
        )

        model = LinearSingleTrack2DOF(p)

        yaw_sensor = YawRateMeasurement(
            float(
                cfg["sensors"]["yaw_rate"]["sigma_radps"]
            )
        )

        ay_cfg = cfg["sensors"]["lateral_acceleration"]
        ay_location = ay_cfg.get("location", "cog")

        if ay_location == "cog":
            ay_sensor = LateralAccelerationMeasurement(
                p,
                float(ay_cfg["sigma_mps2"]),
            )

        elif ay_location == "rear_axle":
            ay_sensor = RearAxleLateralAccelerationMeasurement(
                model,
                float(ay_cfg["sigma_mps2"]),
            )

        else:
            raise ValueError(
                f"Unknown lateral acceleration location: "
                f"{ay_location}"
            )

        sensors = {
            yaw_sensor.source: yaw_sensor,
            ay_sensor.source: ay_sensor,
        }

        dataset = _build_dataset(
            cfg,
            model=model,
            yaw_sensor=yaw_sensor,
            ay_sensor=ay_sensor,
        )

        if estimator_type != "kalman_filter":
            raise ValueError(
                f"Estimator {estimator_type!r} "
                f"is not configured for {model_type!r}"
            )

        ecfg = cfg["estimator"]

        estimator = LinearVehicleKalmanFilter(
            model,
            sensors,
            x0=np.array(
                ecfg["initial_state"],
                dtype=float,
            ),
            P0=np.diag(
                np.array(
                    ecfg["initial_std"],
                    dtype=float,
                )
                ** 2
            ),
            q_std=np.array(
                ecfg["process_noise_std_per_sqrt_s"],
                dtype=float,
            ),
            t0=float(
                ecfg.get("initial_time_s", 0.0)
            ),
            initial_delta=float(
                ecfg.get(
                    "initial_steering_angle_rad",
                    0.0,
                )
            ),
            initial_vx=float(
                ecfg.get(
                    "initial_longitudinal_velocity_mps",
                    10.0,
                )
            ),
        )

        return dataset, model, sensors, estimator

    # =====================================================================
    # 2) JOINT STATE-PARAMETER 2-DoF + EKF
    #    x = [beta, r, Cf, Cr]
    # =====================================================================

    if model_type == "linear_single_track_parameter_estimation":

        if estimator_type not in {
            "extended_kalman_filter_2dof",
            "ekf_2dof_joint",
        }:
            raise ValueError(
                f"Estimator {estimator_type!r} "
                f"is not configured for {model_type!r}"
            )

        # Known vehicle parameters only. Cf and Cr are states of the EKF.
        p = JointSingleTrackParameters(
            **_joint_parameter_kwargs(mp)
        )

        model = JointParametersSingleTrack2DOF(p)

        # --- Sensors -----------------------------------------------------

        yaw_sensor = JointYawRateMeasurement(
            float(
                cfg["sensors"]["yaw_rate"]["sigma_radps"]
            )
        )

        ay_cfg = cfg["sensors"]["lateral_acceleration"]

        # The Golf preprocessor maps acc_y_CAN_ms2 to the generic/CoG-like
        # canonical source, not to the rear-axle source.
        if ay_cfg.get("location", "cog") != "cog":
            raise ValueError(
                "The current joint 2-DoF EKF expects "
                "sensor.lateral_acceleration with location='cog'"
            )

        ay_sensor = JointLateralAccelerationMeasurement(
            model,
            float(ay_cfg["sigma_mps2"]),
        )

        sensors = {
            yaw_sensor.source: yaw_sensor,
            ay_sensor.source: ay_sensor,
        }

        # --- Dataset -----------------------------------------------------

        dataset = _build_dataset(cfg)

        # --- EKF ---------------------------------------------------------

        ecfg = cfg["estimator"]

        initial_state = np.array(
            ecfg["initial_state"],
            dtype=float,
        )

        initial_std = np.array(
            ecfg["initial_std"],
            dtype=float,
        )

        q_std = np.array(
            ecfg["process_noise_std_per_sqrt_s"],
            dtype=float,
        )

        if initial_state.shape != (4,):
            raise ValueError(
                "Joint 2-DoF initial_state must be "
                "[beta, yaw_rate, Cf, Cr]"
            )

        if initial_std.shape != (4,):
            raise ValueError(
                "Joint 2-DoF initial_std must contain 4 values"
            )

        if q_std.shape != (4,):
            raise ValueError(
                "Joint 2-DoF process_noise_std_per_sqrt_s "
                "must contain 4 values"
            )

        estimator = VehicleExtendedKalmanFilter2DOFJoint(
            model,
            sensors,
            x0=initial_state,
            P0=np.diag(initial_std**2),
            q_std=q_std,
            t0=float(
                ecfg.get("initial_time_s", 0.0)
            ),
            initial_delta=float(
                ecfg.get(
                    "initial_steering_angle_rad",
                    0.0,
                )
            ),
            initial_vx=float(
                ecfg.get(
                    "initial_longitudinal_velocity_mps",
                    10.0,
                )
            ),
        )

        return dataset, model, sensors, estimator

    # =====================================================================
    # 2) NONLINEAR 3-DoF + LINEAR LATERAL TIRE LAW + EKF
    # =====================================================================

    if model_type == "nonlinear_single_track_3dof":

        p = NonlinearSingleTrack3DOFLinearTireParameters(
            **_common_parameter_kwargs(mp)
        )

        model = NonlinearSingleTrack3DOFLinearTire(p)

        return _build_3dof_ekf_stack(
            cfg,
            model,
            model_type,
        )

    # =====================================================================
    # 3) NONLINEAR 3-DoF + FIALA TIRE MODEL + EKF
    # =====================================================================

    if model_type == "nonlinear_single_track_3dof_fiala":

        p = NonlinearSingleTrack3DOFFialaParameters(
            **_common_parameter_kwargs(mp),
            road_friction_coefficient=float(
                mp.get(
                    "road_friction_coefficient",
                    1.1,
                )
            ),
            gravity_mps2=float(
                mp.get(
                    "gravity_mps2",
                    9.81,
                )
            ),
        )

        model = NonlinearSingleTrack3DOFFiala(p)

        return _build_3dof_ekf_stack(
            cfg,
            model,
            model_type,
        )

    # =====================================================================
    # UNKNOWN MODEL
    # =====================================================================

    raise ValueError(
        f"Unknown model type: {model_type}"
    )