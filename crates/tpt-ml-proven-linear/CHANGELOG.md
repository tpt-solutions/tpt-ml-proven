# Changelog

All notable changes to `tpt-ml-proven-linear` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Sparse input for `Ridge`, `ElasticNet` and `Lasso` without copying the matrix: the centred
  Gram statistics are built from the non-zeros, and coordinate descent runs on them. Results
  agree with the dense path to about 1e-9 for Ridge and 1e-6 for the coordinate-descent models
  (`tests/sparse.rs`). `Lasso`'s own optimality check still copies the matrix.
- `ArrayView::is_sparse`, which a view uses to say it is stored sparsely (default `false`).
- `Ridge` (closed form), `ElasticNet` and `Lasso` (coordinate descent), matching
  scikit-learn's objectives. Parity tests at 1e-6 on committed fixtures.
- `LogisticRegression`: binary, L2-penalised, fitted by damped Newton iterations with
  scikit-learn's `C` convention. `predict_proba` and `predict`. Parity tests on a committed
  fixture.
- `MulticlassLogistic`: multinomial (softmax) logistic regression for two or more classes,
  same `C` convention. Coefficients and probabilities match scikit-learn on a committed
  fixture. The last class's intercept is pinned at 0, since the objective is invariant to a
  common shift.
- Runtime contract checks for Ridge (normal equations), ElasticNet (KKT conditions at
  convergence), logistic regression (stationarity) and multinomial logistic regression,
  with `ridge.tptc`, `elasticnet.tptc`, `logistic.tptc` and `multiclass-logistic.tptc`.
- Model files (`to_bytes` and `from_bytes`) for Ridge, Lasso, ElasticNet, LogisticRegression
  and MulticlassLogistic, in the version 1 format. A committed Ridge golden file must keep
  loading.

### Tests

- A default-tolerance test for the ElasticNet KKT contract, across penalties and mixing
  ratios.
