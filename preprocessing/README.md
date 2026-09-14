# Preprocessing contract

Raw-source handling stops here. Every raw source gets **one** preprocessor that
converts it to the canonical HDF5 contract. Everything in `src/vehicle_estimation`
uses only canonical signal names, SI units, explicit frames and physical timestamps.

## Canonical HDF5 schema v1.0

```text
run.h5
├── attrs
│   ├── schema_version = "1.0"
│   ├── source_dataset
│   └── source_file
├── signals/
│   ├── input.steering_angle/
│   │   ├── time_s      [N]
│   │   └── value       [N]
│   ├── input.longitudinal_velocity/
│   ├── sensor.yaw_rate/
│   └── sensor.lateral_acceleration_rear_axle/
└── truth/
    └── beta_rad/
        ├── time_s      [M]
        └── value       [M]
```

Each `/signals/<source>` group has attributes:

- `kind`: `input` or `measurement`
- `unit`: SI unit
- `frame`: e.g. `cog`, `rear_axle`, `vehicle`
- `raw_source`: source-specific original signal name

Crucially, **each signal owns its timestamp array**. Do not resample all sensors
onto one common clock in source preprocessing unless that is explicitly part of
an experiment. This preserves future latency/asynchrony/OOSM studies.

## Rule for another dataset

Do not add another estimator-facing `FooDataset` class. Add only:

```text
preprocessing/foo_dataset/raw -> canonical_hdf5
```

Then use the existing `CanonicalHDF5Dataset` unchanged.
