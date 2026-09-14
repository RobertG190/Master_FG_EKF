# Current model: linear 2-DoF single-track model

## State and inputs

```text
x = [beta, r]^T
u = delta
```

- `beta`: vehicle sideslip angle at the CoG [rad]
- `r`: yaw rate [rad/s]
- `delta`: effective road-wheel/single-track steering angle [rad]
- `v_x`: measured longitudinal velocity used as an LPV scheduling variable

## KIT IONIQ 5 parameters

From the uploaded dataset's `parameter.m`:

```text
m   = 2254 kg
Iz  = 2193 kg m^2
lf  = 1.563 m
lr  = 1.437 m
Cf  = 234008 N/rad
Cr  = 234008 N/rad
```

These values are now stored in `configs/vehicles/kit_ioniq5.yaml` rather than
mixed into the experiment configuration.

## Continuous model

```text
x_dot = A(vx) x + B(vx) delta + w
```

with

```text
A11 = -(Cf + Cr)/(m vx)
A12 = -1 - (Cf lf - Cr lr)/(m vx^2)
A21 = -(Cf lf - Cr lr)/Iz
A22 = -(Cf lf^2 + Cr lr^2)/(Iz vx)

B1 = Cf/(m vx)
B2 = Cf lf/Iz
```

The implementation discretizes this exactly for piecewise-constant `vx` and
`delta` using a zero-order-hold augmented matrix exponential.

## Yaw-rate measurement

```text
z_r = [0  1] x + v_r
```

## Lateral acceleration at CoG

```text
a_y,CoG = H_CoG x + D_CoG delta
```

```text
H_CoG = [-(Cf+Cr)/m, -(Cf lf-Cr lr)/(m vx)]
D_CoG = Cf/m
```

## Lateral acceleration at rear axle

The KIT dataset's `a_y_ra_mps2` is not a CoG measurement. With the rear axle at
`x=-lr` relative to the CoG:

```text
a_y,RA = a_y,CoG - lr * r_dot
```

and

```text
r_dot = A_r x + B_r delta
```

therefore

```text
H_RA = H_CoG - lr * A_r
D_RA = D_CoG - lr * B_r
```

This is the measurement equation used by the KIT configuration.

## Validity limitations

This L0 model intentionally has limits that later thesis experiments can expose:

- small tire slip / linear cornering stiffness assumption,
- planar lateral-yaw dynamics only,
- no roll/load transfer,
- no nonlinear friction saturation,
- no longitudinal tire-force coupling,
- `Cf` and `Cr` assumed known and constant,
- poor validity near standstill; `min_vx_mps` prevents numerical singularity but
  does not make the physics valid there.

It is therefore a useful *baseline model*, not the final truth model.
