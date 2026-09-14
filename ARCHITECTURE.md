# Software architecture

## 1. Core rule

**Raw data never enters an estimator.**

Every data source is converted exactly once into a canonical HDF5 run format:

```text
KIT .mat -----------┐
CarMaker export ----┼--> source-specific preprocessor --> canonical .h5
CAN/MDF ------------┤
ROS bag ------------┘
                                                |
                                                v
                                      CanonicalHDF5Dataset
                                                |
                                                v
                                        canonical Events
                                                |
                   +----------------------------+---------------------------+
                   |                            |                           |
                   v                            v                           v
             latency/dropout               estimator                 truth/eval
               scenarios                 KF/EKF/MHE/FG                (separate)
```

A new data source therefore requires **one new preprocessor, not one new
estimator adapter**.

## 2. Why HDF5 and not one wide CSV?

The master thesis explicitly studies asynchrony, latency and sensor failure.
A wide table tends to force all signals onto one common timestamp grid during
preprocessing. That destroys exactly the timing structure later experiments
need.

Canonical HDF5 stores every signal as:

```text
/signals/<canonical_source>/time_s
/signals/<canonical_source>/value
```

Thus IMU at 1000 Hz, GNSS at 10 Hz and delayed CAN signals can coexist without
resampling. HDF5 is also directly writable from MATLAB and Python and scales
better than multi-million-row CSV files.

## 3. Canonical signal contract

For the current 2-DoF KF:

| Canonical source | Kind | Unit | Physical meaning |
|---|---|---:|---|
| `input.steering_angle` | input | rad | effective single-track steering angle |
| `input.longitudinal_velocity` | input | m/s | measured scheduling velocity at CoG |
| `sensor.yaw_rate` | measurement | rad/s | vehicle yaw rate |
| `sensor.lateral_acceleration_rear_axle` | measurement | m/s² | lateral acceleration at rear axle center |
| `truth/beta_rad` | truth only | rad | sideslip reference at CoG |

Each signal group additionally stores `unit`, `frame` and `raw_source` metadata.

Canonical names describe **physics**, not the source dataset. A KIT raw name
such as `w_z_cor_radps` disappears at the preprocessing boundary.

## 4. Ground truth stays separate

Estimator-facing measurements and evaluation reference data are deliberately
separated:

```text
/signals/...  -> estimator
/truth/...    -> evaluation only
```

This prevents accidental leakage of ground truth into the filter and later
makes fair KF/EKF/MHE/factor-graph comparisons easier.

## 5. KIT IONIQ 5 source mapping

The uploaded dataset contains MATLAB `timeseries` objects. Its specific
preprocessor is:

```text
preprocessing/kit_ioniq5/export_kit_ioniq5_to_canonical.m
```

It maps the dynamic runs to the canonical schema and derives a CoG sideslip
reference from Correvit planar velocity, yaw rate and the supplied sensor
translation vectors.

Static vehicle parameters are kept separately in:

```text
configs/vehicles/kit_ioniq5.yaml
```

This is important: vehicle physics is not a property of an experiment file.

## 6. 2-DoF estimation stack

```text
CanonicalHDF5Dataset
        |
        +--> input.steering_angle --------------------+
        +--> input.longitudinal_velocity -------------|----+
        +--> sensor.yaw_rate -------------------------|    |
        +--> sensor.lateral_acceleration_rear_axle ---|----|-->
                                                       v    v
                                            LinearSingleTrack2DOF
                                                       |
                                  +--------------------+--------------------+
                                  |                                         |
                                  v                                         v
                         YawRateMeasurement              RearAxleLateralAcceleration
                                  \                                         /
                                   +------------------+--------------------+
                                                      v
                                         LinearVehicleKalmanFilter
                                                      |
                                                      v
                                             estimates + covariance
```

State:

```text
x = [beta, yaw_rate]^T
```

The longitudinal velocity is a measured scheduling variable, so the model is
LPV rather than globally LTI when `v_x` changes.

## 7. Rear-axle acceleration is modeled at the correct location

The dataset provides `a_y_ra_mps2`, i.e. acceleration at the rear axle center.
The existing CoG acceleration equation is therefore not used directly.

Rigid-body kinematics gives

```text
a_y,RA = a_y,CoG - l_r * r_dot
```

Both terms are linear in `[beta, r, delta]` for the current model. The dedicated
`RearAxleLateralAccelerationMeasurement` therefore remains a linear KF
measurement equation and avoids numerically differentiating noisy yaw-rate
measurements.

## 8. Timing architecture

`Event` contains two times:

```text
measurement_time = when the physical measurement belongs
arrival_time     = when the estimator receives it
```

The canonical data loader initially sets both equal. Scenario transforms later
modify only `arrival_time` for latency/jitter experiments or remove events for
dropout/outage experiments. The original physical timestamp remains intact.

This same event stream can be consumed by:

- baseline KF / EKF,
- OOSM-capable filters,
- MHE,
- fixed-lag factor graphs,
- full/incremental factor graphs.

## 9. Planned extension points

The next estimator/model levels should be added without touching preprocessing:

```text
models/
  linear_single_track.py        # current L0
  nonlinear_single_track.py     # L1
  four_wheel_or_roll_model.py   # L2 estimator model if needed

estimators/
  kalman_filter.py              # current
  ekf.py
  mhe.py
  factor_graph.py

scenarios/
  latency.py
  dropout.py
  outage.py
  bias.py
  noise_scaling.py

architectures/
  centralized.py
  modular_cascade.py
  distributed.py

observability/
  local_observability.py
  fisher_information.py
  excitation_gate.py
```

Before state dimension or parameters grow, add a `StateRegistry` and
`ParameterRegistry` so factor definitions reference semantic variables instead
of hard-coded array indices.

## 10. Reproducible experiment rule

Every run should persist:

- resolved experiment config,
- exact vehicle parameter file,
- input canonical file identifier/hash,
- estimates and covariance,
- diagnostics,
- metrics,
- plots,
- later: git commit and runtime information.

Then model/KF/FG comparisons are changes in configuration/components rather than
separate scripts.
