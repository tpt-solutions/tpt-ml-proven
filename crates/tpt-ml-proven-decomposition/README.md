# tpt-ml-proven-decomposition

Matrix decomposition estimators for [tpt-ml-proven](https://github.com/tpt-solutions/tpt-ml-proven).

> Status: `Pca`, `TruncatedSvd` and `Nmf` are implemented and tested against scikit-learn.

| Estimator | Contract | scikit-learn parity |
| --- | --- | --- |
| `Pca` | [`pca.tptc`](../../docs/contracts/pca.tptc) | components and variances within 1e-9 |
| `TruncatedSvd` | [`svd.tptc`](../../docs/contracts/svd.tptc) | singular values and components (ARPACK) |
| `Nmf` | [`nmf.tptc`](../../docs/contracts/nmf.tptc) | factors within 1e-9 (`init="custom"`, `solver="mu"`) |

## PCA

`Pca` is principal component analysis. Its contract says the components are orthogonal with
unit norm, and the explained variances sum to at most the total variance.

- `fit`, `transform`, `inverse_transform` and `fit_transform`.
- Explained variance, explained variance ratio, total variance, and the feature mean.
- Deterministic sign convention: the largest-magnitude entry of each component is positive.
- Generic over `f32` and `f64`. The eigendecomposition is a cyclic Jacobi solver in this crate.
- Model files: `to_bytes` and `from_bytes`, in the format in
  [docs/model-format.md](../../docs/model-format.md).

```rust
use tpt_ml_proven_core::{DenseView, Transformer, UnsupervisedEstimator};
use tpt_ml_proven_decomposition::Pca;

let data = vec![1.0, 2.0, 2.0, 4.1, 3.0, 5.9, 4.0, 8.0]; // 4 rows x 2 columns

let mut pca = Pca::<f64>::new(1);
pca.fit(DenseView::new(&data, 4, 2).unwrap()).unwrap();
let scores = pca.transform(DenseView::new(&data, 4, 2).unwrap()).unwrap();
```

## TruncatedSVD

`TruncatedSvd` keeps the top singular vectors of the data without centring it. It is dense
only, for now.

## NMF

`Nmf` factorises a non-negative matrix as `W·H`, minimising the Frobenius error with
Lee–Seung multiplicative updates. It uses scikit-learn's stopping rule. Starting factors can be
given with `with_init`, or drawn from a seeded generator with `with_seed`.

Not implemented: scikit-learn's default coordinate-descent solver and its NNDSVD
initialisation.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Licence

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache-2.0](../../LICENSE-APACHE), at
your option.
