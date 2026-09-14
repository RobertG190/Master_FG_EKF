from __future__ import annotations

import numpy as np

from vehicle_estimation.models.linear_single_track import (
    LinearSingleTrack2DOF,
    LinearSingleTrackParameters,
)
from vehicle_estimation.sensors.lateral_acceleration import LateralAccelerationMeasurement
from vehicle_estimation.sensors.rear_axle_lateral_acceleration import (
    RearAxleLateralAccelerationMeasurement,
)


def test_rear_axle_acceleration_matches_rigid_body_relation():
    p = LinearSingleTrackParameters(
        mass_kg=2254.0,
        yaw_inertia_kgm2=2193.0,
        lf_m=1.563,
        lr_m=1.437,
        cornering_stiffness_front_nprad=234008.0,
        cornering_stiffness_rear_nprad=234008.0,
        min_vx_mps=3.0,
    )
    model = LinearSingleTrack2DOF(p)
    cog = LateralAccelerationMeasurement(p, sigma_mps2=0.1)
    rear = RearAxleLateralAccelerationMeasurement(model, sigma_mps2=0.1)

    x = np.array([0.02, 0.15])
    delta = 0.03
    vx = 15.0

    ay_cog = float(cog.predict(x, delta=delta, vx=vx)[0])
    A, B = model.continuous_matrices(vx)
    r_dot = float((A @ x + B[:, 0] * delta)[1])
    expected = ay_cog - p.lr_m * r_dot
    actual = float(rear.predict(x, delta=delta, vx=vx)[0])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
