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
| tpt-ml-proven, current | 2.666 s | 5.065 s | 20 |

Reproduce:

```sh
python tools/parity/bench_kmeans.py                       # scikit-learn, writes the data file
cargo run --release -p tpt-ml-proven-clustering --example bench_kmeans \
    --features parallel -- target/bench/kmeans_1000000x50_f32.bin 1000000 50 8 20
```

## Status against the target

The target is at least 2× faster than scikit-learn on the same hardware. **It is not
met.** On one thread we are about 1.9× slower. With all cores we are about 12× slower.

Changes that helped the single-thread path, from 6.26 s to 2.67 s:

- Distance kernel with eight independent accumulators, so the loop vectorises.
- Row access through `row_slice` where the view is contiguous, instead of per-element
  `get` with bounds checks.
- Contract checks reuse per-cluster ranges gathered during the update, rather than a
  second pass over the data.

Open problems:

- **Parallel path regressed** from 3.50 s to 5.07 s. The only change in the parallel
  path was a single up-front copy of the data (previously it was copied every
  iteration), so the cause is not yet known. Next step: profile the thread-scope and
  join overhead per iteration, and check for contention on the shared labels buffer.
- **Gap to scikit-learn on one thread.** scikit-learn computes distances with a
  matrix-multiply formulation (`|x|² − 2x·c + |c|²`) through BLAS. Matching that
  likely needs either a blocked kernel written here, or a BLAS dependency, which the
  licence rule for runtime dependencies makes awkward.

Results are not yet from CI. The benchmark job is not in `ci.yml`. It is meant to run
as a non-blocking report.
