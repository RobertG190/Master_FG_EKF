from __future__ import annotations

"""
Convert the Golf GTE ETAS/ADMA HDF5 runs to the project-wide canonical HDF5 format.

Raw input (example)
-------------------
001_Asphalt_feucht_Konstantfahrt_30_kmh.h5
    tout
    steer_angle_CAN_rad
    vel_ref_CAN_ms
    yaw_rate_CAN_rads
    acc_y_CAN_ms2
    sideslip_angle_ADMA_rad
    yaw_rate_ADMA_rads
    vel_x_ADMA_ms
    vel_y_ADMA_ms
    ...

Canonical output
----------------
/signals/input.steering_angle/{time_s,value}
/signals/input.longitudinal_velocity/{time_s,value}
/signals/sensor.yaw_rate/{time_s,value}
/signals/sensor.lateral_acceleration/{time_s,value}

/truth/beta_rad/{time_s,value}
/truth/yaw_rate_radps/{time_s,value}
/truth/longitudinal_velocity_mps/{time_s,value}
/truth/lateral_velocity_mps/{time_s,value}

Important modelling decisions
-----------------------------
1. CAN channels are estimator-facing signals.
2. ADMA channels are evaluation/reference channels only.
3. steer_angle_CAN_rad is used directly. Its magnitude is consistent with an
   effective front-road-wheel angle for the single-track model; it is NOT
   divided by veh_steer_ratio.
4. acc_y_CAN_ms2 is stored as sensor.lateral_acceleration, not as
   sensor.lateral_acceleration_rear_axle, because the raw file does not establish
   a rear-axle sensor location.
5. The source files are already resampled to one 50 Hz time grid (tout). The
   canonical format still stores a separate time vector for every signal so
   future datasets may remain asynchronous.
"""

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCHEMA_VERSION = "1.0"

# Raw -> canonical estimator-facing signals.
# source, raw_name, kind, unit, frame, note
SIGNAL_MAP = (
    (
        "input.steering_angle",
        "steer_angle_CAN_rad",
        "input",
        "rad",
        "vehicle",
        "Effective single-track steering input; raw CAN angle used directly.",
    ),
    (
        "input.longitudinal_velocity",
        "vel_ref_CAN_ms",
        "input",
        "m/s",
        "vehicle",
        "CAN vehicle reference velocity; used as known/scheduling longitudinal velocity.",
    ),
    (
        "sensor.yaw_rate",
        "yaw_rate_CAN_rads",
        "measurement",
        "rad/s",
        "vehicle",
        "Vehicle CAN yaw-rate measurement.",
    ),
    (
        "sensor.lateral_acceleration",
        "acc_y_CAN_ms2",
        "measurement",
        "m/s^2",
        "vehicle",
        "Vehicle CAN lateral acceleration. Exact physical sensor location is not documented "
        "in the source HDF5; therefore it is not labelled as rear-axle acceleration.",
    ),
)

# Raw -> canonical evaluation/reference channels.
# state_name, raw_name, unit, frame, note
TRUTH_MAP = (
    (
        "beta_rad",
        "sideslip_angle_ADMA_rad",
        "rad",
        "vehicle",
        "ADMA sideslip-angle reference. Source metadata quality flag is retained at root.",
    ),
    (
        "yaw_rate_radps",
        "yaw_rate_ADMA_rads",
        "rad/s",
        "vehicle",
        "ADMA yaw-rate reference.",
    ),
    (
        "longitudinal_velocity_mps",
        "vel_x_ADMA_ms",
        "m/s",
        "vehicle",
        "ADMA longitudinal-velocity reference.",
    ),
    (
        "lateral_velocity_mps",
        "vel_y_ADMA_ms",
        "m/s",
        "vehicle",
        "ADMA lateral-velocity reference.",
    ),
)

# Useful source channels to preserve for later experiments without forwarding
# them to the current estimator. kind='aux' makes CanonicalHDF5Dataset ignore them.
AUX_MAP = (
    ("aux.can.acceleration_longitudinal", "acc_x_CAN_ms2", "m/s^2", "vehicle"),
    ("aux.adma.acceleration_longitudinal", "acc_x_ADMA_ms2", "m/s^2", "adma"),
    ("aux.adma.acceleration_lateral", "acc_y_ADMA_ms2", "m/s^2", "adma"),
    ("aux.adma.acceleration_lateral_body", "acc_y_body_ADMA_ms2", "m/s^2", "adma_body"),
    ("aux.adma.acceleration_longitudinal_body", "acc_x_body_ADMA_ms2", "m/s^2", "adma_body"),
    ("aux.adma.acceleration_vertical_body", "acc_z_body_ADMA_ms2", "m/s^2", "adma_body"),
    ("aux.can.wheel_velocity_fl", "wheel_vel_fl_CAN_ms", "m/s", "wheel_fl"),
    ("aux.can.wheel_velocity_fr", "wheel_vel_fr_CAN_ms", "m/s", "wheel_fr"),
    ("aux.can.wheel_velocity_rl", "wheel_vel_rl_CAN_ms", "m/s", "wheel_rl"),
    ("aux.can.wheel_velocity_rr", "wheel_vel_rr_CAN_ms", "m/s", "wheel_rr"),
    ("aux.can.brake_pressure", "brake_pressure_CAN_bar", "bar", "vehicle"),
    ("aux.can.brake_torque", "brake_torque_CAN_Nm", "N*m", "vehicle"),
    ("aux.can.drive_torque", "drive_torque_CAN_Nm", "N*m", "vehicle"),
    ("aux.gnss.speed", "vel_gnss_ADMA_ms", "m/s", "gnss"),
    ("aux.gnss.sats_used", "gnss_sats_used", "1", "gnss"),
    ("aux.gnss.sats_visible", "gnss_sats_visible", "1", "gnss"),
    ("aux.gnss.mode", "gnss_mode", "1", "gnss"),
    ("aux.quality.lateral_stimulated", "kf_lat_stimulated", "1", "source"),
    ("aux.quality.longitudinal_stimulated", "kf_long_stimulated", "1", "source"),
    ("aux.quality.steady_state", "kf_steady_state", "1", "source"),
)


def _as_1d_float(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name!r} is empty")
    return arr


def _validate_time(t: np.ndarray) -> None:
    if not np.all(np.isfinite(t)):
        raise ValueError("tout contains non-finite values")
    if len(t) > 1 and np.any(np.diff(t) <= 0.0):
        raise ValueError("tout must be strictly increasing")


def _read_raw_signal(
    src: h5py.File,
    raw_name: str,
    t: np.ndarray,
    *,
    required: bool = True,
) -> np.ndarray | None:
    if raw_name not in src:
        if required:
            raise KeyError(f"Required raw signal {raw_name!r} is missing")
        return None

    y = _as_1d_float(src[raw_name][...], name=raw_name)
    if len(y) != len(t):
        raise ValueError(
            f"{raw_name!r}: length {len(y)} does not match tout length {len(t)}"
        )
    return y


def _write_signal(
    dst: h5py.File,
    *,
    source: str,
    t: np.ndarray,
    y: np.ndarray,
    kind: str,
    unit: str,
    frame: str,
    raw_source: str,
    note: str | None = None,
) -> None:
    base = dst.require_group(f"signals/{source}")
    base.create_dataset("time_s", data=t.astype(np.float64), compression="gzip")
    base.create_dataset("value", data=y.astype(np.float64), compression="gzip")

    base.attrs["kind"] = kind
    base.attrs["unit"] = unit
    base.attrs["frame"] = frame
    base.attrs["raw_source"] = raw_source
    if note:
        base.attrs["note"] = note


def _write_truth(
    dst: h5py.File,
    *,
    state_name: str,
    t: np.ndarray,
    y: np.ndarray,
    unit: str,
    frame: str,
    raw_source: str,
    note: str | None = None,
) -> None:
    base = dst.require_group(f"truth/{state_name}")
    base.create_dataset("time_s", data=t.astype(np.float64), compression="gzip")
    base.create_dataset("value", data=y.astype(np.float64), compression="gzip")

    base.attrs["unit"] = unit
    base.attrs["frame"] = frame
    base.attrs["raw_source"] = raw_source
    if note:
        base.attrs["note"] = note


def _copy_source_attrs(src: h5py.File, dst: h5py.File) -> None:
    """Retain raw-file metadata for provenance without changing its meaning."""
    for key, value in src.attrs.items():
        target = f"source_{key}"
        try:
            dst.attrs[target] = value
        except (TypeError, ValueError):
            # Fallback for uncommon attribute types.
            dst.attrs[target] = str(value)


def convert_file(raw_path: str | Path, output_path: str | Path) -> Path:
    raw_path = Path(raw_path)
    output_path = Path(output_path)

    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(raw_path, "r") as src:
        if "tout" not in src:
            raise KeyError(f"{raw_path.name}: required time signal 'tout' is missing")

        t = _as_1d_float(src["tout"][...], name="tout")
        _validate_time(t)

        # Validate all estimator-facing and truth channels before creating output.
        signal_values = {
            raw_name: _read_raw_signal(src, raw_name, t, required=True)
            for _, raw_name, *_ in SIGNAL_MAP
        }
        truth_values = {
            raw_name: _read_raw_signal(src, raw_name, t, required=True)
            for _, raw_name, *_ in TRUTH_MAP
        }

        # Avoid leaving a half-written file after a failed conversion.
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        try:
            with h5py.File(tmp_path, "w") as dst:
                # Canonical contract / provenance.
                dst.attrs["schema_version"] = SCHEMA_VERSION
                dst.attrs["source_dataset"] = "Golf GTE ETAS/ADMA"
                dst.attrs["source_file"] = raw_path.name
                dst.attrs["coordinate_convention"] = (
                    "Source convention retained; verify against dataset documentation "
                    "before frame-sensitive model extensions."
                )

                if len(t) > 1:
                    dt = float(np.median(np.diff(t)))
                    dst.attrs["nominal_sample_rate_hz"] = 1.0 / dt
                else:
                    dst.attrs["nominal_sample_rate_hz"] = np.nan

                dst.attrs["preprocessor"] = Path(__file__).name
                dst.attrs["steering_mapping"] = (
                    "steer_angle_CAN_rad -> input.steering_angle, no steering-ratio division"
                )
                dst.attrs["lateral_acceleration_mapping"] = (
                    "acc_y_CAN_ms2 -> sensor.lateral_acceleration; exact sensor location "
                    "not established by source HDF5"
                )

                _copy_source_attrs(src, dst)

                # Estimator-facing signals.
                for source, raw_name, kind, unit, frame, note in SIGNAL_MAP:
                    _write_signal(
                        dst,
                        source=source,
                        t=t,
                        y=signal_values[raw_name],
                        kind=kind,
                        unit=unit,
                        frame=frame,
                        raw_source=raw_name,
                        note=note,
                    )

                # Evaluation/reference channels.
                for state_name, raw_name, unit, frame, note in TRUTH_MAP:
                    _write_truth(
                        dst,
                        state_name=state_name,
                        t=t,
                        y=truth_values[raw_name],
                        unit=unit,
                        frame=frame,
                        raw_source=raw_name,
                        note=note,
                    )

                # Optional useful channels for future work.
                for source, raw_name, unit, frame in AUX_MAP:
                    y = _read_raw_signal(src, raw_name, t, required=False)
                    if y is None:
                        continue
                    _write_signal(
                        dst,
                        source=source,
                        t=t,
                        y=y,
                        kind="aux",
                        unit=unit,
                        frame=frame,
                        raw_source=raw_name,
                    )

            tmp_path.replace(output_path)

        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    return output_path


def convert_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    pattern: str = "*.h5",
) -> list[Path]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching {pattern!r} found in {input_dir}"
        )

    outputs: list[Path] = []
    for raw_path in files:
        out_path = output_dir / raw_path.name
        result = convert_file(raw_path, out_path)
        outputs.append(result)
        print(f"converted: {raw_path.name} -> {result}")

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Golf GTE ETAS/ADMA HDF5 runs to canonical HDF5."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing the raw Golf GTE .h5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/golf_gte"),
        help="Directory for canonical .h5 files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.h5",
        help="Input file glob pattern (default: *.h5).",
    )
    args = parser.parse_args()

    convert_directory(
        args.input_dir,
        args.output_dir,
        pattern=args.pattern,
    )


if __name__ == "__main__":
    main()