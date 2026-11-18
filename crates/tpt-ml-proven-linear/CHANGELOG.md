# Changelog

All notable changes to `tpt-ml-proven-linear` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Runtime contract checks for Ridge, ElasticNet (including the KKT conditions at convergence)
  and logistic regression, with `ridge.tptc`, `elasticnet.tptc` and `logistic.tptc`.

### Added

- `Ridge` (closed form), `ElasticNet` and `Lasso` (coordinate descent), matching
  scikit-learn's objectives. Parity tests at 1e-6 on committed fixtures.
- `LogisticRegression`: binary, L2-penalised, fitted by damped Newton iterations with
  scikit-learn's `C` convention. `predict_proba` and `predict`. Parity tests on a committed
  fixture. Multiclass is not yet supported.
