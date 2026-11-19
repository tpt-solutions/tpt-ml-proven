# Changelog

All notable changes to `tpt-ml-proven-trees` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `DecisionTreeRegressor`: CART regression with exact greedy splits, scikit-learn's midpoint
  thresholds, and a `min_samples_leaf` limit.
- `GradientBoostingRegressor`: squared-error boosting over regression trees, with a learning
  rate in `(0, 1]`, `train_loss` per round, and access to the fitted trees.
- Runtime contract checks: leaves partition the rows, depth is bounded, and each boosting
  round does not increase the training error (`decision-tree.tptc`, `gradient-boosting.tptc`).
- Parity: gradient-boosting predictions match scikit-learn 1.9.0 to 1e-9 on the
  `gbm_regression` fixture.
- Model files (`to_bytes` and `from_bytes`) for both estimators, in the version 1 format.
- Integration tests for the invariants, and rejection of out-of-range parameters.
