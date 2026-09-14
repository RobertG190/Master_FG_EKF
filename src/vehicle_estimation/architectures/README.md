# Future estimator compositions

Keep estimator algorithm and estimator architecture distinct.

Planned compositions:

- `CentralEstimator`: one joint latent state/problem,
- `CascadeEstimator`: outputs of one estimator are inputs of another,
- `DistributedEstimator`: multiple local estimators with explicit information exchange.

A factor graph can be physically modular while still statistically central/joint.
