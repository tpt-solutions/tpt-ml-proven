# tpt-ml-proven — Project Todo

Maintainer: TPT Solutions · License: MIT OR Apache-2.0 · Home: github.com/tpt-solutions/tpt-ml-proven

Baselines: latest stable Rust · Python 3.9+ (abi3 wheels) · Go 1.21+ · CI on GitHub Actions
Dependency rule: runtime deps must be MIT, MIT/Apache-2.0 dual, BSD, ISC, Zlib, Unlicense or CC0. **No Apache-2.0-only dependencies.** Build/dev tools may use any OSS license.

---

## Phase 0 — Project setup & governance
- [x] Initialise git repo and `.gitignore`
- [x] Create Cargo workspace (`Cargo.toml`) with `crates/` layout from spec §7
- [x] Add `LICENSE-MIT` and `LICENSE-APACHE` (copyright: TPT Solutions)
- [x] Set `license = "MIT OR Apache-2.0"` and authors/repository metadata in every crate
- [x] Add SPDX headers (`MIT OR Apache-2.0`) convention to source files (documented in `CONTRIBUTING.md`, applied to every `lib.rs`)
- [x] Add `deny.toml` (cargo-deny): license allowlist, ban Apache-2.0-only crates, advisories check
- [x] Add `rustfmt.toml`, clippy config, `rust-toolchain.toml` (stable)
- [x] GitHub Actions skeleton: fmt, clippy, test, cargo-deny, on Linux/macOS/Windows (`.github/workflows/ci.yml`; not yet run on GitHub)
- [x] Add README stub, CONTRIBUTING.md (issues only)
- [x] Copy spec into `docs/DESIGN.md`; create `docs/contracts/` directory
- [x] Audit the dependency tree of tpt-math-*, tpt-formal and tpt-telos for Apache-only licenses (record result: `docs/dependency-audit.md`)
- [x] Add `ffi/python/` and `ffi/go/` placeholder directories

**Done when:** empty workspace builds, CI is green, cargo-deny passes.

## Phase 1 — Core crate (`tpt-ml-proven-core`)
- [x] Crate scaffold: `#![no_std]` + `alloc`, `std` feature flag
- [x] `ArrayView` trait: zero-copy over borrowed/owned row-major data (`DenseView`, `Dense`). Adapters for tpt-math-linalg-dense / -sparse are deferred to the algorithm crates so core stays dependency-free
- [x] `TargetView` trait for labels/targets (impl for `&[T]`)
- [x] Generic float abstraction supporting `f32` and `f64` (own `Float` trait, no `libm` dependency)
- [x] `SupervisedEstimator` trait (`fit` returns `Result`)
- [x] `UnsupervisedEstimator` trait
- [x] `Predictor` trait (`predict` returns `Result`)
- [x] `Transformer` trait (`transform`, `fit_transform`)
- [x] Shared error types (fit errors, shape errors, contract violations)
- [x] Contract glue: macros for `requires!` / `ensures!` (attribute form deferred)
- [x] Runtime assertion mode for data-dependent contracts (cheap checks, toggleable via `runtime-contracts` feature)
- [x] Compile-time checks for static shapes / constant parameters (`static_require!`, works with const generics)
- [x] Composition via generics/impl blocks (no runtime Pipeline object) — design + example (crate docs and `tests/traits.rs`)
- [x] Unit tests, docs, doc-tests

**Done when:** core compiles under `no_std + alloc`, traits documented and exercised by a toy estimator. ✅ (`ColumnCenterer` and `MeanRegressor` in `tpt-ml-proven-core::toy`)

## Phase 2 — Verification infrastructure
- [ ] Integrate tpt-formal contract language and the self-contained QF_LRA solver (no Z3) — partial: QF_LRA is `tpt-solver` (git dep in `tools/tptc-verify`, pinned rev). The `.tptc` syntax is our own; `tpt-formal`'s contract crate is not used yet
- [x] Define `.tptc` file format conventions and naming (one per algorithm in `docs/contracts/`; see `docs/contracts/README.md`)
- [x] Contract-to-code linkage: map each `.tptc` clause to a runtime assertion or static check (`"name.id"` literal, checked by `tptc-verify`)
- [x] CI job that solver-verifies every `.tptc` file (`contracts` job in `.github/workflows/ci.yml`; needs tpt-solver pushed to GitHub first)
- [x] Negative tests: deliberately false contracts must fail verification (`tools/tptc-verify/tests/verify.rs`)
- [x] Developer guide: how to write and verify a contract (`docs/contracts/README.md`)

**Done when:** CI fails on any unverified or broken contract and passes on the sample contracts.

## Phase 3 — KMeans (`tpt-ml-proven-clustering`)
- [x] Crate scaffold depending only on core (no tpt-math leaves needed for KMeans)
- [x] Lloyd iteration (dense input)
- [x] Sparse input support (`CsrView` in core; `get` is a binary search, so it is correct but not yet optimised)
- [x] Initialisation: k-means++ and random; seeded RNG for reproducibility (SplitMix64)
- [x] Multi-threaded Lloyd iteration (`parallel` feature; core stays no_std). Copies dense data for the threads. Results match sequential exactly
- [x] Convergence criteria (tolerance, max iterations), empty-cluster handling (keeps previous centre)
- [x] `fit` / `predict` / `transform` via core traits; `FitError` variants (core `Error`: NonFinite, InvalidParameter, EmptyInput, NotFitted, Shape)
- [x] Generic over `f32` and `f64`
- [x] `docs/contracts/kmeans.tptc`: requires (rows ≥ 1, k ≥ 1, k ≤ rows); ensures (monotonic inertia descent, convergence ⇒ stable assignments, centres in convex hull)
- [x] Runtime assertions for data-dependent clauses (`kmeans.*` names in `kmeans.rs`; linkage passes)
- [x] Unit + property tests (edge cases: k = 1, k = n, duplicate points, NaN/inf rejection)

**Done when:** KMeans works on dense and sparse data and its contracts verify in CI. ✅ locally (CI contracts job not yet run on GitHub)

## Phase 4 — PCA (`tpt-ml-proven-decomposition`)
- [x] Crate scaffold depending only on core (no tpt-math leaves)
- [x] Eigensolver: own cyclic Jacobi decomposition (`eigen.rs`) in place of tpt-math-linalg-dense, so the crate has no external dependency. Computation is in f64
- [x] Centering, component selection, explained variance (+ ratio), total variance
- [x] `fit` / `transform` / `inverse_transform` / `fit_transform`
- [x] Deterministic sign convention for components (largest-magnitude entry positive)
- [x] Generic over `f32` and `f64`
- [x] `docs/contracts/pca.tptc`: requires (rows ≥ 2, 1 ≤ n_components ≤ min(rows, cols)); ensures (orthogonality, unit norm, Σ explained_variance ≤ total_variance)
- [x] Runtime assertions with numerical tolerance for orthogonality/unit-norm (1e-8, in f64 before rounding)
- [x] Unit + property tests (rank-deficient data, constant columns, n_components bounds)

**Done when:** PCA contracts verify in CI and results are stable across platforms.

## Phase 5 — scikit-learn parity & benchmarks
- [x] Pinned generator script (offline, uses a fixed scikit-learn version) producing KMeans and PCA reference vectors (`tools/parity/generate_fixtures.py`, pinned to 1.9.0)
- [x] Commit fixtures (so CI never needs scikit-learn for correctness) (`tools/parity/fixtures/`)
- [x] Parity tests in `f64` within 1e-9 tolerance (KMeans, PCA)
- [x] Document unavoidable differences (init RNG, sign flips) and how tests normalise them (`docs/parity.md`)
- [x] Benchmark harness: KMeans.fit on 1M × 50 `f32` (`examples/bench_kmeans.rs`)
- [x] Comparison script against scikit-learn on same hardware (`tools/parity/bench_kmeans.py`). Target ≥ 2× faster: **not met** (see `docs/benchmarks.md`)
- [x] CI benchmark job as non-blocking report (`benchmark` job in `ci.yml`: `continue-on-error`, results in the job summary and an artifact). Not yet run on GitHub; results still to be copied into `docs/`
- [ ] Profile and optimise if below 2× target — done: single-thread 6.26 s → 2.63 s, 20 threads 0.73 s (parallel update chunked). Still 1.9× slower single-threaded and 1.7× slower with all cores vs scikit-learn (see `docs/benchmarks.md`)

**Done when:** parity tests pass at 1e-9 and benchmark results are published.

## Phase 6 — FFI surface (`tpt-ml-proven-ffi`, `ffi/`)
- [ ] ~~Generate bindings through the tpt-telos FFI bridge~~ — superseded: the bridge rejects float, array and generic types at the FFI boundary (only integers are supported), so bindings are generated per language by standard tools (PyO3/maturin, wasm-bindgen, cbindgen for cgo). Decision recorded in `docs/bindings.md`
- [x] **Python:** PyO3 + maturin, abi3 wheels for Python 3.9+ (built as `cp39-abi3`); scikit-learn-style API, with `fit` returning self (`KMeans(n_clusters=8).fit(X).predict(X)`)
- [x] Python: zero-copy NumPy input via `__array_interface__` (the buffer protocol is not in the stable ABI); GIL released during `fit`
- [ ] Python: wheels for Linux, macOS, Windows installable without a C compiler (test in clean containers/VMs). Windows build works locally; Linux/macOS and clean-VM installs not yet tested
- [ ] **Go:** cgo package at `ffi/go` (module `github.com/tpt-solutions/tpt-ml-proven/ffi/go`), Go 1.21+, idiomatic API (`NewKMeans`, `Fit`, `Predict`)
- [ ] Go: prebuilt static libs / headers so builds need no cgo gymnastics
- [x] **WASM:** wasm-bindgen / wasm-pack build exposing KMeans and PCA (`ffi/wasm`, Node target)
- [x] Binding tests for Python (12) and WASM (7), including f32/f64 and error propagation. Go: not yet (see below)
- [x] Usage examples for Python and WASM (`ffi/python/README.md`, `ffi/wasm/README.md`). Go: not yet
- [ ] CI jobs building and testing all three bindings. Added: Python wheel on Linux, macOS and Windows (pytest); WASM (Node tests); C ABI (tests and header check). Go job pending the Go package. Go is blocked locally: no C compiler on this machine and `CGO_ENABLED=0`

**Done when:** KMeans and PCA run from Python, Go and WASM in CI on all three OSes.

## Phase 7 — v0.1 release
- [ ] README with one-paragraph verification story
- [ ] API docs (rustdoc), usage guide, contract docs
- [ ] CHANGELOG
- [ ] Success-criteria sign-off (spec §10):
  - [ ] KMeans and PCA match scikit-learn vectors within 1e-9
  - [ ] All `docs/contracts/` solver-verified in CI
  - [ ] Python wheel installs on Linux/macOS/Windows with no C compiler
  - [ ] Go package builds without cgo header gymnastics
  - [ ] KMeans 1M × 50 f32 ≥ 2× faster than scikit-learn
  - [ ] README explains verification in one paragraph
- [ ] Final cargo-deny license audit (no Apache-only deps)
- [ ] Publish crates to crates.io (core → algorithm crates → ffi)
- [ ] Publish wheels to PyPI (trusted publishing)
- [ ] Tag `v0.1.0` and create GitHub release (Go module picks up the tag)

**Done when:** v0.1.0 is installable from crates.io, PyPI and via `go get`.

## Phase 8 — Roadmap: remaining algorithms
Each item gets: crate code, `.tptc` contract, scikit-learn parity fixtures, FFI exposure, tests. Done so far: crate code, parity fixtures and tests, and `.tptc` contracts with runtime checks. Not yet: FFI exposure for these algorithms.

### Clustering
- [x] DBSCAN — exact label parity with scikit-learn (`dbscan_blobs_noise`)
- [x] HDBSCAN — excess-of-mass selection; cluster partition and noise match scikit-learn (`hdbscan_blobs_noise`)

### Decomposition
- [x] TruncatedSVD — singular values and components match scikit-learn ARPACK (`svd_arpack_k2`). Dense only so far
- [x] NMF — Frobenius loss, multiplicative updates; matches scikit-learn with `init="custom"`, `solver="mu"` (`nmf_mu_fixed_iters`, `nmf_mu_tol`). scikit-learn's default `cd` solver and NNDSVD init are not implemented

### Linear models (`tpt-ml-proven-linear`)
- [x] Ridge — closed form, matches scikit-learn (`linear_ridge`)
- [x] Lasso — coordinate descent, matches scikit-learn (`linear_lasso`)
- [x] ElasticNet — coordinate descent, matches scikit-learn (`linear_elasticnet`)
- [x] Logistic regression — binary, L2, Newton iterations; matches scikit-learn (`logistic_binary_c1`)
- [x] Logistic regression: multiclass (multinomial) — `MulticlassLogistic`, matches scikit-learn coefficients and probabilities (`logistic_multiclass_c1`)

### Preprocessing (`tpt-ml-proven-preprocessing`)
- [x] Scalers (Standard, MinMax, Robust) — unit tests with hand-computed values; scikit-learn parity at 1e-9 (`scalers_skewed`)
- [x] Encoders (OneHot, Ordinal) — unit tests; unknown categories are an error
- [x] Imputers (Simple: mean, median, most frequent, constant) — unit tests

### Ensemble
- [x] Stump forests (`tpt-ml-proven-ensemble`) — bagged depth-one classification trees, majority vote. No scikit-learn reference, so tested by properties

- [ ] Update README/docs and release as v0.x increments — per-crate CHANGELOGs updated for linear and preprocessing; root README and CHANGELOG not yet updated for these

## Phase 9 — v0.2
- [ ] Design stable on-disk model format (replace interim bincode/serde)
- [ ] Interim: serde/bincode support for models behind a feature flag (v0.1.x)
- [ ] Format versioning and backward-compatibility tests
- [ ] Go vanity import path `tptsolutions.co.nz/mlproven` (go-import meta tags, module path migration)
- [ ] Documentation site

## Phase 10 — Future / out of scope for now
- [ ] `tpt-ml-proven-trees` crate: gradient boosting / random forests with structural-invariant verification
- [ ] GPU backends (after tpt-gpu is stable)
- [ ] npm publishing for the WASM package (optional)
