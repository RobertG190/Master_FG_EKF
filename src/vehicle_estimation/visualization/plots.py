from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_state_plots(estimates: pd.DataFrame, truth: pd.DataFrame, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if estimates.empty or truth.empty:
        return

    est = estimates.sort_values("time").groupby("time", as_index=False).tail(1)

    for name, var_name, ylabel, filename in [
        ("beta_rad", "var_beta", r"$\beta$ [rad]", "beta.png"),
        ("yaw_rate_radps", "var_yaw_rate", r"$r$ [rad/s]", "yaw_rate.png"),
        ("longitudinal_velocity_mps", "var_longitudinal_velocity", r"$v_x$ [m/s]", "vx.png"),
        ("lateral_velocity_mps", "var_lateral_velocity", r"$v_y$ [m/s]", "vy.png"),
    ]:
        if name not in truth.columns or name not in est.columns:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(truth["time"], truth[name], label="truth")
        ax.plot(est["time"], est[name], label="estimate")
        if var_name in est.columns:
            s = 2.0 * np.sqrt(np.maximum(est[var_name].to_numpy(), 0.0))
            ax.fill_between(est["time"], est[name] - s, est[name] + s, alpha=0.2, label="±2σ")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / filename, dpi=160)
        plt.close(fig)
