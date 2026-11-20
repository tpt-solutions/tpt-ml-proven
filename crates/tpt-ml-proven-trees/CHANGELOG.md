# Changelog

All notable changes to `tpt-ml-proven-trees` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `DecisionTreeRegressor` and `DecisionTreeClassifier`: `min_samples_split`, a `max_leaf_nodes`
  limit (the tree then grows best-first, as scikit-learn does), and `feature_importances()`, each
  feature's share of the impurity reduction. Model files use the new kinds 26 and 27 when they
  store these; the older kinds 8 and 19 still load.
- `RandomForestRegressor`, `RandomForestClassifier` and `GradientBoostingRegressor` take the same
  `min_samples_split` and `max_leaf_nodes` for each tree, and report `feature_importances()`. The
  forests average the trees' shares, as scikit-learn does; boosting normalises the squared-error
  reductions summed over all rounds. Their files use kinds 28 to 30 when they store these.
  Boosting's importances agree with scikit-learn's to about 0.002 on the test data but not
  exactly; the cause is not yet traced.
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
