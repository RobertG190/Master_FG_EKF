import numpy as np

from vehicle_estimation.models.linear_single_track import LinearSingleTrack2DOF, LinearSingleTrackParameters


def model():
    return LinearSingleTrack2DOF(
        LinearSingleTrackParameters(1500.0, 2500.0, 1.2, 1.6, 80000.0, 80000.0, 2.0)
    )


def test_shapes_and_discretization():
    m = model()
    A, B = m.continuous_matrices(20.0)
    F, Bd = m.transition_matrices(delta=0.01, vx=20.0, dt=0.01)
    assert A.shape == (2, 2)
    assert B.shape == (2, 1)
    assert F.shape == (2, 2)
    assert Bd.shape == (2, 1)
    assert np.all(np.isfinite(F))


def test_continuous_model_is_stable_at_nominal_speed():
    A, _ = model().continuous_matrices(20.0)
    eig = np.linalg.eigvals(A)
    assert np.all(np.real(eig) < 0.0)
