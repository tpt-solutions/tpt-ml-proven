# Benchmarks

KMeans `fit` on 1,000,000 samples × 50 features, `f32`, 8 clusters, 20 Lloyd
iterations, `tol = 0`, the same starting centres in both implementations (the first
8 rows). Time is the best of three runs. Data is standard normal, seed 7, written
once to `target/bench/` (git-ignored) so both sides read identical bytes.

Machine: 20 logical cores, Windows. Release builds only (`cargo run --release`).

| Implementation | 1 thread | 20 threads | Iterations |
| --- | --- | --- | --- |
| scikit-learn 1.9.0 (`algorithm="lloyd"`) | 1.389 s | 0.426 s | 20 |
| tpt-ml-proven, first version | 6.257 s | 3.499 s | 20 |
| tpt-ml-proven, parallel update | 2.634 s | 0.725 s | 20 |

Thread scaling for tpt-ml-proven (same data and settings):

| Threads | 1 | 2 | 4 | 8 | 20 |
| --- | --- | --- | --- | --- | --- |
| Seconds | 2.634 | 1.714 | 1.059 | 0.837 | 0.725 |

Reproduce:

```sh
python tools/parity/bench_kmeans.py                       # scikit-learn, writes the data file
cargo run --release -p tpt-ml-proven-clustering --example bench_kmeans \
    --features parallel -- target/bench/kmeans_1000000x50_f32.bin 1000000 50 8 20
```

## Status against the target

The target is at least 2× faster than scikit-learn on the same hardware. **It is not
met.** On one thread we are about 1.9× slower. With all cores we are about 1.7× slower.

Changes that helped the single-thread path, from 6.26 s to 2.67 s:

- Distance kernel with eight independent accumulators, so the loop vectorises.
- Row access through `row_slice` where the view is contiguous, instead of per-element
  `get` with bounds checks.
- Contract checks reuse per-cluster ranges gathered during the update, rather than a
  second pass over the data.

Open problems:

- **Parallel update was the bottleneck.** The centre update ran on one thread and
  scanned all the data each iteration, which capped the speed-up. It is now chunked
  (4096 rows per chunk) and merged in chunk order, so results are identical for any
  thread count. This took the 20-thread time from 2.06 s to 0.72 s. The earlier 5.07 s
  figure was not reproduced on a second run, so treat it as noise.
- **Remaining all-core gap** of about 1.7×, and the single-thread gap of about 1.9×.
  The threads now scale well up to 8, and flatten after that.
- **Gap to scikit-learn on one thread.** scikit-learn computes distances with a
  matrix-multiply formulation (`|x|² − 2x·c + |c|²`) through BLAS. Matching that
  likely needs either a blocked kernel written here, or a BLAS dependency, which the
  licence rule for runtime dependencies makes awkward.

Results are not yet from CI. The benchmark job is not in `ci.yml`. It is meant to run
as a non-blocking report.
