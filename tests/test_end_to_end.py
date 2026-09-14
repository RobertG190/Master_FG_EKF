from pathlib import Path

from vehicle_estimation.core.config import load_yaml
from vehicle_estimation.evaluation.metrics import evaluate
from vehicle_estimation.experiments.factory import build_stack


def test_synthetic_kf_end_to_end():
    root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(root / "configs" / "synthetic_kf.yaml")
    dataset, _, _, estimator = build_stack(cfg)
    events, truth = dataset.load()
    for event in events:
        estimator.process(event)
    metrics = evaluate(estimator.estimates_frame(), truth)
    assert metrics["rmse_beta_rad"] < 0.02
    assert metrics["rmse_yaw_rate_radps"] < 0.01
    assert estimator.diagnostics()["rejected_oosm"] == 0
