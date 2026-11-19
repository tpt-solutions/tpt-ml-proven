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
- `persist` module: the version 1 model file format, with a magic header, a format version, a
  model kind, little-endian fields, and an FNV-1a checksum. Readers accept versions up to the
  current one, refuse newer files, check the kind, and bound every length by the bytes left.
  The specification is in [docs/model-format.md](../../docs/model-format.md).
- Model kinds for every model in the workspace, with stable numbers (`persist::kind`).
- `Error::Format`, for model files that cannot be read. `Error` is `non_exhaustive`, so
  existing matches are not affected.
