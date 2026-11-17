# Changelog

All notable changes to `tpt-ml-proven-core` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Estimator traits: `SupervisedEstimator`, `UnsupervisedEstimator`, `Predictor` and
  `Transformer`.
- Dense and CSR input views (`ArrayView`, `TargetView`, `DenseView`, `Dense`, `CsrView`).
- The `Float` trait for `f32` and `f64`.
- The shared `Error`, `FitError`, `ShapeError` and `ContractViolation` types.
- The `requires!`, `ensures!` and `static_require!` contract macros.
- `toy` module with `ColumnCenterer` and `MeanRegressor`, used to exercise the traits.
- `no_std` + `alloc` support, with `std` and `runtime-contracts` features.
