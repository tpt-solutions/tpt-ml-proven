# tpt-ml-proven-clustering

Clustering estimators for [tpt-ml-proven](https://github.com/tpt-solutions/tpt-ml-proven).

> Status: `KMeans`, `Dbscan` and `Hdbscan` are implemented and tested against scikit-learn.

| Estimator | Contract | scikit-learn parity |
| --- | --- | --- |
| `KMeans` | [`kmeans.tptc`](../../docs/contracts/kmeans.tptc) | centres and labels within 1e-9 on a fixture |
| `Dbscan` | [`dbscan.tptc`](../../docs/contracts/dbscan.tptc) | exact labels on a fixture |
| `Hdbscan` | [`hdbscan.tptc`](../../docs/contracts/hdbscan.tptc) | same partition and noise points on a fixture |

## KMeans

`KMeans` runs Lloyd's algorithm with k-means++ or random initialisation. Its contract says
inertia never increases, a converged run has stable assignments, and each centre lies in the
convex hull of its points.

- Dense and sparse (`CsrView`) input.
- Seeded and reproducible (`with_seed`). Explicit starting centres: `with_initial_centers`.
- Convergence: `with_tol` and `with_max_iter`. Empty clusters keep their previous centre.
- Generic over `f32` and `f64`.
- Optional `parallel` feature: a multi-threaded assignment step. Results match the sequential
  path exactly, for any thread count. The core crate stays `no_std`.
- Model files: `to_bytes` and `from_bytes`, in the format in
  [docs/model-format.md](../../docs/model-format.md).

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

## DBSCAN

`Dbscan` labels dense regions as clusters and everything else as noise (`-1`). Core samples
have at least `min_samples` neighbours within `eps`, counting themselves.

```rust
use tpt_ml_proven_clustering::Dbscan;
use tpt_ml_proven_core::UnsupervisedEstimator;

let mut db = Dbscan::<f64>::new(0.5, 3);
db.fit(DenseView::new(&data, 4, 2).unwrap()).unwrap();
let labels = db.labels().unwrap();          // &[i64], -1 for noise
```

## HDBSCAN

`Hdbscan` builds the mutual-reachability minimum spanning tree, condenses the single-linkage
tree with `min_cluster_size`, and selects clusters by excess of mass. It has no `eps`, so
clusters of different density are found. The algorithm follows scikit-learn.

Memory is O(n²), because the pairwise distance matrix is held in memory.

## Performance

On the benchmark problem (1M × 50 `f32`), KMeans is slower than scikit-learn: about 1.9× on
one thread and about 1.7× with all cores. See [docs/benchmarks.md](../../docs/benchmarks.md)
for the method, and run the benchmark with:

```sh
cargo run --release -p tpt-ml-proven-clustering --example bench_kmeans \
    --features parallel -- <data.bin> <rows> <cols> <k> <iters>
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Licence

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache-2.0](../../LICENSE-APACHE), at
your option.
