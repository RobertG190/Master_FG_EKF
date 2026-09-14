import numpy as np

from vehicle_estimation.core.types import Event, EventKind
from vehicle_estimation.estimators.extended_kalman_filter_3dof import VehicleExtendedKalmanFilter3DOF
from vehicle_estimation.models.nonlinear_single_track_3dof import (
    NonlinearSingleTrack3DOF,
    NonlinearSingleTrack3DOFParameters,
)
from vehicle_estimation.sensors.nonlinear_3dof import (
    LongitudinalVelocityMeasurement3DOF,
    RearAxleLateralAccelerationMeasurement3DOF,
    YawRateMeasurement3DOF,
)


def _model():
    return NonlinearSingleTrack3DOF(
        NonlinearSingleTrack3DOFParameters(
            mass_kg=2254.0,
            yaw_inertia_kgm2=2193.0,
            lf_m=1.563,
            lr_m=1.437,
            cornering_stiffness_front_nprad=234008.0,
            cornering_stiffness_rear_nprad=234008.0,
            min_vx_mps=3.0,
        )
    )


def test_rear_axle_ax_enters_longitudinal_dynamics():
    model = _model()
    x = np.array([20.0, 0.5, 0.2])
    xdot = model.continuous_dynamics(
        x, delta=0.0, ax_rear_axle_mps2=1.0
    )
    expected = 1.0 - model.p.lr_m * 0.2**2 + 0.2 * 0.5
    assert np.isclose(xdot[0], expected)


def test_ekf_processes_3dof_event_stream():
    model = _model()
    sensors = {
        s.source: s
        for s in [
            LongitudinalVelocityMeasurement3DOF(0.05),
            YawRateMeasurement3DOF(0.005),
            RearAxleLateralAccelerationMeasurement3DOF(model, 0.10),
        ]
    }
    ekf = VehicleExtendedKalmanFilter3DOF(
        model,
        sensors,
        x0=np.array([20.0, 0.0, 0.0]),
        P0=np.diag([1.0, 0.5, 0.05]) ** 2,
        q_std=np.array([0.3, 0.3, 0.05]),
    )

    # First input values at t=0.
    ekf.process(Event(0.0, 0.0, EventKind.INPUT, "input.steering_angle", np.array([0.02])))
    ekf.process(Event(0.0, 0.0, EventKind.INPUT, "input.longitudinal_acceleration_rear_axle", np.array([0.0])))

    truth = np.array([20.0, 0.0, 0.0])
    for k in range(1, 21):
        t = 0.01 * k
        truth = model.discrete_dynamics(
            truth, delta=0.02, ax_rear_axle_mps2=0.0, dt=0.01
        )
        measurements = [
            ("sensor.longitudinal_velocity", truth[0]),
            ("sensor.yaw_rate", truth[2]),
            (
                "sensor.lateral_acceleration_rear_axle",
                model.lateral_acceleration_rear_axle(truth, delta=0.02),
            ),
        ]
        for source, value in measurements:
            ekf.process(Event(t, t, EventKind.MEASUREMENT, source, np.array([value])))

    frame = ekf.estimates_frame()
    assert not frame.empty
    assert {"longitudinal_velocity_mps", "lateral_velocity_mps", "yaw_rate_radps", "beta_rad"}.issubset(frame.columns)
    assert np.all(np.isfinite(frame[["longitudinal_velocity_mps", "lateral_velocity_mps", "yaw_rate_radps", "beta_rad"]].to_numpy()))
    assert ekf.diagnostics()["measurement_updates"] == 60
