from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from vehicle_estimation.core.config import load_yaml, serializable_copy
from vehicle_estimation.evaluation.metrics import evaluate
from vehicle_estimation.experiments.factory import build_stack
from vehicle_estimation.visualization.plots import save_state_plots


def run_experiment(config_path: str) -> Path:
    cfg = load_yaml(config_path)
    dataset, _, _, estimator = build_stack(cfg)
    events, truth = dataset.load()

    # Arrival time is the runner clock. Stable list order handles equal timestamps.
    events = sorted(enumerate(events), key=lambda kv: (kv[1].arrival_time, kv[0]))
    for _, event in events:
        estimator.process(event)

    estimates = estimator.estimates_frame()
    metrics = evaluate(estimates, truth)

    out_dir = Path(cfg["run"].get("output_root", "results")) / cfg["run"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    estimates.to_csv(out_dir / "estimates.csv", index=False)
    truth.to_csv(out_dir / "truth.csv", index=False)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (out_dir / "diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(estimator.diagnostics(), f, indent=2)
    with (out_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(serializable_copy(cfg), f, sort_keys=False)

    save_state_plots(estimates, truth, out_dir / "plots")

    print(json.dumps(metrics, indent=2))
    print(f"Results: {out_dir.resolve()}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
