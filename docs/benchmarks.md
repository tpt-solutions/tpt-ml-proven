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
| tpt-ml-proven, profiled and tuned | 1.452 s | 0.601 s | 20 |
| tpt-ml-proven, fused pass | 1.611 s | 0.420 s | 20 |

Thread scaling for the fused pass: 1.61 s, 0.97 s, 0.62 s, 0.47 s and 0.42 s at 1, 2,
4, 8 and 20 threads.

The fused pass assigns each sample and accumulates the next centres in the same sweep,
so the data is read once per iteration instead of twice. That costs a little on one
thread, where the work is compute-bound, and helps a lot on many threads, where memory
bandwidth is shared. The all-core time is now level with scikit-learn.

Tried and rejected: an AVX2 distance kernel chosen at run time, with the same
arithmetic as the portable code, so results would be bit-identical. It was slower on
one thread (1.80 to 2.09 s), because the portable code already compiles to good SSE2
and the per-centre reduction costs as much as the wider lanes save. It was faster only
marginally on all cores. It was removed.

Thread scaling for the parallel-update version, before the fused pass (same data and settings):

| Threads | 1 | 2 | 4 | 8 | 20 |
| --- | --- | --- | --- | --- | --- |
| Seconds | 2.634 | 1.714 | 1.059 | 0.837 | 0.725 |

Reproduce:

```sh
python tools/parity/bench_kmeans.py                       # scikit-learn, writes the data file
cargo run --release -p tpt-ml-proven-clustering --example bench_kmeans \
    --features parallel -- target/bench/kmeans_1000000x50_f32.bin 1000000 50 8 20
```

## Profile of the single-thread path

Measured by temporary timers around each phase of the fit loop (one thread, 20
iterations, before the tuning below):

| Phase | Time |
| --- | --- |
| Assignment (distances to all centres) | 1.00 s |
| Centre update (sums, ranges) | 1.30 s |
| Contract checks and the rest | about 0.1 s |

The update was the surprise. Its accumulator indexed each element through bounds
checks and branched on the range, so it did not vectorise. Rewriting it over slices
with branch-free min and max cut the update to 0.39 s and the total to 1.43 s. A run
without the timers gives 1.45 s.

Tried and rejected: Hamerly's exact pruning (upper and lower bounds on distances, so
most assignments need no full search). On this benchmark it was slower, 2.08 s on one
thread. Random 50-dimensional data has distances that concentrate, so the bounds
rarely prove an assignment, and the extra inertia pass costs more than it saves. The
change was reverted. It may help on clustered data, which is not the benchmark here.

The build targets baseline x86-64 (SSE2), while scikit-learn's BLAS uses wider vectors.
That is the likeliest cause of the single-thread gap.

## Status against the target

The target is at least 2× faster than scikit-learn on the same hardware. **It is not
met.** On one thread we are about 1.16× slower (1.61 s against 1.39 s). With all cores we
are level (0.42 s against 0.43 s), which is about 1.0× and not 2×.

Changes that helped the single-thread path, from 6.26 s to 1.45 s (before the fused pass):

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
- **Remaining gap:** about 1.16× slower on one thread (1.61 s against 1.39 s). With all
  cores the fused pass is level with scikit-learn (0.42 s against 0.43 s). Reaching 2× would
  need a kernel faster than scikit-learn's BLAS, and the AVX2 attempt above did not manage
  that. The threads scale well up to 8, and flatten after that.
- **Gap to scikit-learn on one thread.** scikit-learn computes distances with a
  matrix-multiply formulation (`|x|² − 2x·c + |c|²`) through BLAS. Matching that
  likely needs either a blocked kernel written here, or a BLAS dependency, which the
  licence rule for runtime dependencies makes awkward.

The benchmark job runs in `ci.yml` as a non-blocking report, with results in the job
summary and as an artifact. It has not yet run on GitHub, so the figures above come from a
local Windows machine.
