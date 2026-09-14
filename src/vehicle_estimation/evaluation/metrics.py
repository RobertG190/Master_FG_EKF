from __future__ import annotations

import numpy as np
import pandas as pd


def _last_estimate_per_time(estimates: pd.DataFrame) -> pd.DataFrame:
    if estimates.empty:
        return estimates
    return estimates.sort_values("time").groupby("time", as_index=False).tail(1)


def evaluate(estimates: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float]:
    est = _last_estimate_per_time(estimates)
    if est.empty or truth.empty:
        return {}

    metrics: dict[str, float] = {}
    t_est = est["time"].to_numpy(dtype=float)
    t_true = truth["time"].to_numpy(dtype=float)

    for name in ["beta_rad", "yaw_rate_radps", "longitudinal_velocity_mps", "lateral_velocity_mps"]:
        if name not in est.columns or name not in truth.columns:
            continue
        y_est = est[name].to_numpy(dtype=float)
        y_true = np.interp(t_est, t_true, truth[name].to_numpy(dtype=float))
        err = y_est - y_true
        metrics[f"rmse_{name}"] = float(np.sqrt(np.mean(err**2)))
        metrics[f"mae_{name}"] = float(np.mean(np.abs(err)))
        metrics[f"max_abs_{name}"] = float(np.max(np.abs(err)))

    return metrics
