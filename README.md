# Vehicle State Estimation Starter v2

Research-oriented starter architecture for comparing KF/EKF/MHE/factor graphs
on vehicle state-estimation problems without coupling estimators to a particular
raw dataset.

## Main architectural decision

```text
raw source -> one source-specific preprocessor -> canonical HDF5 -> all estimators
```

A future dataset should **not** get another estimator-facing loader. Only write a
new preprocessor that maps its names, units, frames and timestamps to the
canonical contract in `preprocessing/README.md`.

This matters especially for latency/asynchrony research because every canonical
signal keeps its own physical timestamp.

## Current stack

- linear/LPV 2-DoF single-track model, `x=[beta,r]`
- event-driven Kalman filter
- yaw-rate measurement
- CoG or rear-axle lateral-acceleration measurement model
- canonical HDF5 data interface
- separate truth path for evaluation
- scenario interface for future latency/dropout/OOSM experiments
- synthetic sanity-check dataset
- KIT IONIQ 5 source preprocessor and vehicle parameters

## KIT IONIQ 5 dataset

The uploaded raw dataset stores signals as MATLAB `timeseries` objects. Convert
it once with:

```matlab
export_kit_ioniq5_to_canonical( ...
    "C:\path\to\raw\data\dataset", ...
    "C:\path\to\this_project\data\processed\kit_ioniq5");
```

The MATLAB exporter is in:

```text
preprocessing/kit_ioniq5/export_kit_ioniq5_to_canonical.m
```

Then run the Python estimator from the repository root:

```bash
python -m pip install -e .
python -m vehicle_estimation.experiments.run --config configs/kit_ioniq5_kf.yaml
```

The example configuration uses `dynamic_driving_asphalt_a_3.h5` and decimates
the source's 1000 Hz streams by `stride: 10` at load time for a 100 Hz baseline.
The canonical file itself remains full rate.

## Synthetic implementation check

```bash
python -m vehicle_estimation.experiments.run --config configs/synthetic_kf.yaml
pytest -q
```

The synthetic test intentionally uses the same 2-DoF physics for truth and
estimator. Good RMSE there validates code paths and formulas, **not** scientific
model quality.

## Repository structure

```text
vehicle_state_estimation_starter_v2/
├── configs/
│   ├── kit_ioniq5_kf.yaml
│   ├── synthetic_kf.yaml
│   └── vehicles/
│       └── kit_ioniq5.yaml
├── preprocessing/
│   ├── README.md
│   └── kit_ioniq5/
│       ├── README.md
│       └── export_kit_ioniq5_to_canonical.m
├── src/vehicle_estimation/
│   ├── core/
│   ├── data/
│   │   ├── canonical_hdf5.py
│   │   └── synthetic.py
│   ├── models/
│   ├── sensors/
│   ├── estimators/
│   ├── scenarios/
│   ├── observability/
│   ├── factors/
│   ├── architectures/
│   ├── evaluation/
│   ├── visualization/
│   └── experiments/
├── tests/
├── ARCHITECTURE.md
└── MODEL.md
```

See `ARCHITECTURE.md` for the intended path toward nonlinear models, parameter
estimation, centralized/modular/distributed architectures, MHE and factor graphs.

## 3-DoF EKF (KIT IONIQ 5)

The 3-DoF experiment uses state `x=[vx, vy, r]`, steering and measured rear-axle
longitudinal acceleration as inputs, and `vx`, yaw rate and rear-axle lateral
acceleration as measurements.

Because these channels were added after the first 2-DoF export, regenerate the
canonical KIT HDF5 files once with the current MATLAB preprocessor. Then run:

```bash
python -m vehicle_estimation.experiments.run --config configs/kit_ioniq5_3dof_ekf.yaml
```

The existing 2-DoF KF still uses `configs/kit_ioniq5_kf.yaml` unchanged.
