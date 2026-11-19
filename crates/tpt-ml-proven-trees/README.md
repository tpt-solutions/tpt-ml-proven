# tpt-ml-proven-trees

Tree models for [tpt-ml-proven](https://github.com/tpt-solutions/tpt-ml-proven): CART
regression trees and gradient boosting. Their structural invariants are contracts.

> Status: `DecisionTreeRegressor` and `GradientBoostingRegressor` are implemented. The
> boosting predictions match scikit-learn's `GradientBoostingRegressor` to 1e-9 on a
> committed fixture.

| Estimator | Contract | scikit-learn parity |
| --- | --- | --- |
| `DecisionTreeRegressor` | [`decision-tree.tptc`](../../docs/contracts/decision-tree.tptc) | invariant tests |
| `GradientBoostingRegressor` | [`gradient-boosting.tptc`](../../docs/contracts/gradient-boosting.tptc) | predictions within 1e-9 |

## Structural invariants

Three properties hold for every fitted tree and every boosting run. Each is a contract:

1. **The leaves partition the training rows.** Every row lands in exactly one leaf, so the
   leaf counts add up to the number of rows.
2. **Depth is bounded by `max_depth`.** No path from the root is longer.
3. **A boosting round never increases the training squared error.** A leaf moves to the mean
   residual of its rows, and a step of size `learning_rate ∈ (0, 1]` towards that optimum
   cannot increase a convex quadratic.

The contracts are checked at runtime inside the crate, and the verifier checks their linkage
to the source.

## Trees

`DecisionTreeRegressor` grows a CART tree with exact greedy splits. Each split minimises the
squared error of its two children. The threshold is the midpoint of two adjacent distinct
values, as scikit-learn chooses it. A leaf predicts the mean of its targets.

```rust
use tpt_ml_proven_core::{DenseView, Predictor, SupervisedEstimator};
use tpt_ml_proven_trees::DecisionTreeRegressor;

let mut tree = DecisionTreeRegressor::new(3).with_min_samples_leaf(2);
tree.fit(DenseView::new(&x, rows, cols).unwrap(), &y[..]).unwrap();
let predictions = tree.predict(DenseView::new(&x, rows, cols).unwrap()).unwrap();
```

## Gradient boosting

`GradientBoostingRegressor` starts from the mean of the targets. Each round fits a tree to the
current residuals and adds `learning_rate` times its predictions. It uses squared-error loss,
no subsampling and all features at every split, so the fit is deterministic.

```rust
use tpt_ml_proven_trees::GradientBoostingRegressor;

let mut model = GradientBoostingRegressor::new(100, 3)
    .with_learning_rate(0.1)
    .with_min_samples_leaf(2);
model.fit(DenseView::new(&x, rows, cols).unwrap(), &y[..]).unwrap();
let curve = model.train_loss().unwrap();   // mean squared error after each round
```

## Not yet

- Classification trees and random forests.
- Subsampling and feature subsampling.
- Missing-value handling.

## Model files

Both estimators have `to_bytes` and `from_bytes`, in the format in
[docs/model-format.md](../../docs/model-format.md).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Licence

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache-2.0](../../LICENSE-APACHE), at
your option.
