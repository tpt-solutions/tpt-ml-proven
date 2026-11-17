# tpt-ml-proven-decomposition

Matrix decomposition estimators for [tpt-ml-proven](https://github.com/tpt-solutions/tpt-ml-proven).

> Status: `Pca` is implemented. TruncatedSVD and NMF are planned.

## PCA

`Pca` is principal component analysis. Its contract is
[`docs/contracts/pca.tptc`](../../docs/contracts/pca.tptc): the components are
orthogonal with unit norm, and the explained variances sum to at most the total variance.

- `fit`, `transform`, `inverse_transform` and `fit_transform`.
- Explained variance, explained variance ratio, total variance, and the feature mean.
- Deterministic sign convention: the largest-magnitude entry of each component is positive.
- Generic over `f32` and `f64`.
- The eigendecomposition is a cyclic Jacobi solver written in this crate (`eigen.rs`),
  computed in `f64`. The crate has no dependencies beyond the core crate.

```rust
use tpt_ml_proven_core::{DenseView, Transformer, UnsupervisedEstimator};
use tpt_ml_proven_decomposition::Pca;

let data = vec![1.0, 2.0, 2.0, 4.1, 3.0, 5.9, 4.0, 8.0]; // 4 rows x 2 columns

let mut pca = Pca::<f64>::new(1);
pca.fit(DenseView::new(&data, 4, 2).unwrap()).unwrap();

let scores = pca.transform(DenseView::new(&data, 4, 2).unwrap()).unwrap();
println!("explained variance ratio: {:?}", pca.explained_variance_ratio());
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Licence

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache-2.0](../../LICENSE-APACHE), at
your option.
