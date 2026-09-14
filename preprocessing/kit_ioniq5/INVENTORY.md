# Uploaded KIT dataset inventory

The archive contains 41 MAT measurement runs plus `parameter.m`.

For the current 2-DoF state estimator the useful raw signals are:

```text
delta_stm_rad    effective single-track steering angle
w_z_cor_radps    yaw rate
v_x_cor_mps      Correvit longitudinal velocity
v_y_cor_mps      Correvit lateral velocity
a_y_ra_mps2      lateral acceleration at rear axle center
```

The MAT files use several signal subsets depending on maneuver. The dynamic
runs containing the complete set needed by the current baseline are exported by
the provided MATLAB preprocessor; standstill/parking runs are intentionally not
forced into the 2-DoF estimator because the linear bicycle model is not a useful
low-speed model there.

Static parameters in `parameter.m`:

```text
wheelbase      3.000 m
mass           2254 kg
yaw inertia    2193 kg m^2
lf             1.563 m
lr             1.437 m
Cf             234008 N/rad
Cr             234008 N/rad
tire radius    0.36 m
mu asphalt_a   1.1
mu cobblestone 0.7
```

Some dynamic MAT files also expose fields named `v_cog_mps` and
`beta_cog_mps`. Because the latter name's unit suffix is inconsistent with a
slip angle, the starter does not depend on that field. It derives the CoG
sideslip reference from Correvit planar velocity, yaw rate and the supplied
sensor translation vectors instead.
