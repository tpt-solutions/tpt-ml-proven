# Changelog

All notable changes to `tpt-ml-proven-clustering` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `KMeans` (Lloyd's algorithm) with k-means++ and random initialisation, seeded and
  reproducible, for dense and sparse input. Empty clusters keep their previous centre.
- Optional `parallel` feature: a multi-threaded assignment step whose results match the
  sequential path exactly.
- `Dbscan`, with labels matching scikit-learn on a committed fixture.
- `Hdbscan`: excess-of-mass selection, following scikit-learn. Partition and noise match on a
  committed fixture (`hdbscan_blobs_noise`). O(n²) memory.
- Runtime contract checks for DBSCAN (`dbscan.tptc`) and HDBSCAN (`hdbscan.tptc`), and the
  KMeans contract checks.
- Model files for `KMeans` (`to_bytes` and `from_bytes`), with a committed version 1 golden
  file that must keep loading.
- Benchmark example `examples/bench_kmeans.rs` for 1M × 50 `f32`, with a scikit-learn
  comparison.

### Known limitations

- KMeans is about 1.9× slower than scikit-learn on one thread and about 1.7× slower with all
  cores on the benchmark problem (see `docs/benchmarks.md`).
- HDBSCAN uses O(n²) memory and does not yet produce membership probabilities.
