from __future__ import annotations

import h5py
import numpy as np

from vehicle_estimation.core.types import EventKind
from vehicle_estimation.data.canonical_hdf5 import CanonicalHDF5Dataset


def _write_signal(h5, source: str, kind: str, t, y):
    g = h5.require_group(f"signals/{source}")
    g.create_dataset("time_s", data=np.asarray(t, dtype=float))
    g.create_dataset("value", data=np.asarray(y, dtype=float))
    g.attrs["kind"] = kind
    g.attrs["unit"] = "test"


def test_canonical_hdf5_preserves_independent_timestamps(tmp_path):
    path = tmp_path / "run.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["schema_version"] = "1.0"
        _write_signal(h5, "input.steering_angle", "input", [0.0, 0.1], [0.0, 0.01])
        _write_signal(h5, "input.longitudinal_velocity", "input", [0.0, 0.1], [10.0, 10.1])
        _write_signal(h5, "sensor.yaw_rate", "measurement", [0.02, 0.12], [0.1, 0.2])
        _write_signal(
            h5,
            "sensor.lateral_acceleration_rear_axle",
            "measurement",
            [0.03, 0.13],
            [0.5, 0.6],
        )
        tg = h5.require_group("truth/beta_rad")
        tg.create_dataset("time_s", data=[0.0, 0.1])
        tg.create_dataset("value", data=[0.01, 0.02])

    events, truth = CanonicalHDF5Dataset(str(path)).load()
    assert len(events) == 8
    assert events[0].kind == EventKind.INPUT
    assert [e.measurement_time for e in events] == sorted(e.measurement_time for e in events)
    assert any(abs(e.measurement_time - 0.03) < 1e-12 for e in events)
    assert list(truth.columns) == ["time", "beta_rad"]
    np.testing.assert_allclose(truth["beta_rad"], [0.01, 0.02])
