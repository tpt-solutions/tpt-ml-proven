# Changelog

All notable changes to `tpt-ml-proven-decomposition` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `Pca`, with explained variance and explained variance ratio, total variance and the feature
  mean. `fit`, `transform`, `inverse_transform` and `fit_transform`. Deterministic sign
  convention (largest-magnitude entry positive). Eigendecomposition by a cyclic Jacobi solver
  written in the crate, so the crate has no dependencies beyond the core crate.
- `TruncatedSvd` (dense), matching scikit-learn's ARPACK singular values and components.
- `Nmf`: non-negative factorisation with Frobenius loss and multiplicative updates. Matches
  scikit-learn with `init="custom"` and `solver="mu"` at 1e-9. The default `cd` solver and
  NNDSVD initialisation are not implemented.
- Runtime contract checks for PCA, TruncatedSVD (`svd.tptc`) and NMF (`nmf.tptc`).
- Model files for `Pca` (`to_bytes` and `from_bytes`), with a committed version 1 golden file.
