from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Top-level YAML config must be a mapping.")
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    required = ["run", "dataset", "model", "estimator", "sensors"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")

    model_type = cfg["model"].get("type")
    estimator_type = cfg["estimator"].get("type")
    supported_pairs = {
        ("linear_single_track_2dof", "kalman_filter"),
        ("nonlinear_single_track_3dof", "extended_kalman_filter_3dof"),
        ("nonlinear_single_track_3dof", "ekf_3dof"),
        ("nonlinear_single_track_3dof_fiala", "extended_kalman_filter_3dof"),
        ("linear_single_track_parameter_estimation", "extended_kalman_filter_2dof")
    }
    if (model_type, estimator_type) not in supported_pairs:
        raise ValueError(
            f"Unsupported model/estimator combination: {model_type!r} / {estimator_type!r}. "
            f"Supported: {sorted(supported_pairs)}"
        )


def serializable_copy(cfg: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(cfg)
