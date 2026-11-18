# Changelog

All notable changes to `tpt-ml-proven-clustering` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `Hdbscan`: excess-of-mass cluster selection, following scikit-learn. Partition and noise
  match scikit-learn on a committed fixture (`hdbscan_blobs_noise`). O(n²) memory.
- Runtime contract checks for DBSCAN (`dbscan.tptc`) and HDBSCAN (`hdbscan.tptc`).

### Added

- `KMeans` (Lloyd's algorithm) with k-means++ and random initialisation, seeded and
  reproducible, for dense and sparse input.
- Empty-cluster handling: an empty cluster keeps its previous centre.
- Optional `parallel` feature: a multi-threaded assignment step whose results match the
  sequential path exactly.
- Benchmark example `examples/bench_kmeans.rs` for 1M × 50 `f32`, with a scikit-learn
  comparison.

### Known limitations

- KMeans is about 1.9× slower than scikit-learn on one thread and about 1.7× slower with
  all cores on the benchmark problem (see `docs/benchmarks.md`).
