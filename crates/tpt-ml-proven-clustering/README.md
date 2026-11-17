# tpt-ml-proven-clustering

Clustering estimators for [tpt-ml-proven](https://github.com/tpt-solutions/tpt-ml-proven).

> Status: `KMeans` is implemented. DBSCAN and HDBSCAN are planned.

## KMeans

`KMeans` runs Lloyd's algorithm with k-means++ or random initialisation. Its contract is
[`docs/contracts/kmeans.tptc`](../../docs/contracts/kmeans.tptc): inertia never increases,
a converged run has stable assignments, and each centre lies in the convex hull of its
points.

- Dense and sparse (`CsrView`) input.
- Seeded and reproducible (`with_seed`). Explicit starting centres are available with
  `with_initial_centers`.
- Convergence is controlled by `with_tol` and `with_max_iter`. Empty clusters keep their
  previous centre.
- Generic over `f32` and `f64`.
- Optional `parallel` feature: a multi-threaded assignment step. Its results match the
  sequential path exactly, for any thread count. The core crate stays `no_std`.

```rust
use tpt_ml_proven_clustering::{Init, KMeans};
use tpt_ml_proven_core::{DenseView, Predictor, UnsupervisedEstimator};

let data = vec![0.0, 0.0, 0.1, 0.0, 10.0, 10.0, 10.1, 10.0]; // 4 rows x 2 columns

let mut km = KMeans::<f64>::new(2)
    .with_init(Init::KMeansPlusPlus)
    .with_seed(42);
km.fit(DenseView::new(&data, 4, 2).unwrap()).unwrap();

let labels = km.predict(DenseView::new(&data, 4, 2).unwrap()).unwrap();
assert_eq!(labels[0], labels[1]);
assert_ne!(labels[0], labels[2]);
```

### Performance

On the benchmark problem (1M × 50 `f32`), KMeans is slower than scikit-learn: about 1.9×
on one thread and about 1.7× with all cores. See [docs/benchmarks.md](../../docs/benchmarks.md)
for the method. The benchmark is `examples/bench_kmeans.rs`:

```sh
cargo run --release -p tpt-ml-proven-clustering --example bench_kmeans \
    --features parallel -- <data.bin> <rows> <cols> <k> <iters>
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Licence

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache-2.0](../../LICENSE-APACHE), at
your option.
