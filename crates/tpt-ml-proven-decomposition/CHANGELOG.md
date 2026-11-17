# Changelog

All notable changes to `tpt-ml-proven-decomposition` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `Pca` with explained variance and explained variance ratio, total variance, and the
  feature mean.
- `fit`, `transform`, `inverse_transform` and `fit_transform`.
- Deterministic sign convention for components (largest-magnitude entry positive).
- Eigendecomposition by a cyclic Jacobi solver written in the crate (`eigen.rs`), so the
  crate has no dependencies beyond the core crate.
