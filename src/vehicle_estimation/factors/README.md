# Future factor library

Do not duplicate the vehicle equations here. Factor wrappers should call the same
`DynamicModel` and `MeasurementModel` objects used by KF/EKF.

Planned groups:

- dynamics factors,
- yaw-rate measurement factor,
- lateral-acceleration factor,
- parameter prior/random-walk/hold factors,
- robust noise models,
- excitation-aware factor policies.
