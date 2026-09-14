import numpy as np

from vehicle_estimation.models.linear_single_track import LinearSingleTrackParameters
from vehicle_estimation.sensors.lateral_acceleration import LateralAccelerationMeasurement
from vehicle_estimation.sensors.yaw_rate import YawRateMeasurement


def params():
    return LinearSingleTrackParameters(1500.0, 2500.0, 1.2, 1.6, 80000.0, 80000.0, 2.0)


def test_measurement_models():
    x = np.array([0.01, 0.1])
    yaw = YawRateMeasurement(0.01)
    ay = LateralAccelerationMeasurement(params(), 0.1)
    assert np.isclose(yaw.predict(x, delta=0.02, vx=20.0)[0], 0.1)
    assert yaw.jacobian_x(x, delta=0.02, vx=20.0).shape == (1, 2)
    assert ay.jacobian_x(x, delta=0.02, vx=20.0).shape == (1, 2)
    assert np.isfinite(ay.predict(x, delta=0.02, vx=20.0)[0])
