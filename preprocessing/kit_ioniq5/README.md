# KIT IONIQ 5 preprocessor

Source: **Multi-Surface Driving Maneuvers: A Dataset from Standstill to Vehicle
Dynamics Limits**, DOI `10.35097/44a91t97pmnha1k9`.

The source MAT files contain MATLAB `timeseries` MCOS objects. Therefore the
one-time source conversion is intentionally MATLAB, while the estimator stack
remains pure Python.

## Export

In MATLAB, add this folder to the path and run:

```matlab
export_kit_ioniq5_to_canonical( ...
    "C:\path\to\raw\data\dataset", ...
    "C:\path\to\vehicle_state_estimation_starter_v2\data\processed\kit_ioniq5");
```

The exporter currently writes dynamic runs that contain all signals needed for
`x=[beta,r]` 2-DoF estimation.

### Mapping used

| Raw signal | Canonical signal | Use |
|---|---|---|
| `delta_stm_rad` | `input.steering_angle` | model input |
| `w_z_cor_radps` | `sensor.yaw_rate` | KF measurement |
| `a_y_ra_mps2` | `sensor.lateral_acceleration_rear_axle` | KF measurement |
| `v_x_cor_mps` + geometry + yaw rate | `input.longitudinal_velocity` | measured scheduling variable at CoG |
| `v_x_cor_mps`, `v_y_cor_mps`, yaw rate + geometry | `truth/beta_rad` | Correvit-derived CoG sideslip reference |

The original files are synchronous at 1000 Hz, but the canonical format still
retains a timestamp per signal. This is deliberate preparation for later
asynchrony/latency experiments.
