# tpt-ml-proven — Project Todo

Maintainer: TPT Solutions · License: MIT OR Apache-2.0 · Home: github.com/tpt-solutions/tpt-ml-proven

Baselines: latest stable Rust · Python 3.9+ (abi3 wheels) · Go 1.21+ · CI on GitHub Actions
Dependency rule: runtime deps must be MIT, MIT/Apache-2.0 dual, BSD, ISC, Zlib, Unlicense or CC0. **No Apache-2.0-only dependencies.** Build/dev tools may use any OSS license.

---

## Phase 0 — Project setup & governance
- [ ] Initialise git repo and `.gitignore`
- [ ] Create Cargo workspace (`Cargo.toml`) with `crates/` layout from spec §7
- [ ] Add `LICENSE-MIT` and `LICENSE-APACHE` (copyright: TPT Solutions)
- [ ] Set `license = "MIT OR Apache-2.0"` and authors/repository metadata in every crate
- [ ] Add SPDX headers (`MIT OR Apache-2.0`) convention to source files
- [ ] Add `deny.toml` (cargo-deny): license allowlist, ban Apache-2.0-only crates, advisories check
- [ ] Add `rustfmt.toml`, clippy config, `rust-toolchain.toml` (stable)
- [ ] GitHub Actions skeleton: fmt, clippy, test, cargo-deny, on Linux/macOS/Windows
- [ ] Add README stub, CONTRIBUTING.md (issues only)
- [ ] Copy spec into `docs/DESIGN.md`; create `docs/contracts/` directory
- [ ] Audit the dependency tree of tpt-math-*, tpt-formal and tpt-telos for Apache-only licenses (record result)
- [ ] Add `ffi/python/` and `ffi/go/` placeholder directories

**Done when:** empty workspace builds, CI is green, cargo-deny passes.

## Phase 1 — Core crate (`tpt-ml-proven-core`)
- [ ] Crate scaffold: `#![no_std]` + `alloc`, `std` feature flag
- [ ] `ArrayView` trait: zero-copy over tpt-math-linalg-dense and tpt-math-linalg-sparse
- [ ] `TargetView` trait for labels/targets
- [ ] Generic float abstraction supporting `f32` and `f64`
- [ ] `SupervisedEstimator` trait (`fit` returns `Result`)
- [ ] `UnsupervisedEstimator` trait
- [ ] `Predictor` trait
- [ ] `Transformer` trait (`transform`, `fit_transform`)
- [ ] Shared error types (fit errors, shape errors, contract violations)
- [ ] Contract glue: macros/attributes for `requires` / `ensures`
- [ ] Runtime assertion mode for data-dependent contracts (cheap checks, toggleable)
- [ ] Compile-time checks for static shapes / constant parameters
- [ ] Composition via generics/impl blocks (no runtime Pipeline object) — design + example
- [ ] Unit tests, docs, doc-tests

**Done when:** core compiles under `no_std + alloc`, traits documented and exercised by a toy estimator.

## Phase 2 — Verification infrastructure
- [ ] Integrate tpt-formal contract language and the self-contained QF_LRA solver (no Z3)
- [ ] Define `.tptc` file format conventions and naming (one per algorithm in `docs/contracts/`)
- [ ] Contract-to-code linkage: map each `.tptc` clause to a runtime assertion or static check
- [ ] CI job that solver-verifies every `.tptc` file
- [ ] Negative tests: deliberately false contracts must fail verification
- [ ] Developer guide: how to write and verify a contract

**Done when:** CI fails on any unverified or broken contract and passes on the sample contracts.

## Phase 3 — KMeans (`tpt-ml-proven-clustering`)
- [ ] Crate scaffold depending only on core + needed tpt-math leaves
- [ ] Lloyd iteration (dense input)
- [ ] Sparse input support
- [ ] Initialisation: k-means++ and random; seeded RNG for reproducibility
- [ ] Multi-threaded Lloyd iteration (feature-gated so core stays no_std)
- [ ] Convergence criteria (tolerance, max iterations), empty-cluster handling
- [ ] `fit` / `predict` / `transform` via core traits; `FitError` variants
- [ ] Generic over `f32` and `f64`
- [ ] `docs/contracts/kmeans.tptc`: requires (rows ≥ 1, k ≥ 1, k ≤ rows); ensures (monotonic inertia descent, convergence ⇒ stable assignments, centres in convex hull)
- [ ] Runtime assertions for data-dependent clauses
- [ ] Unit + property tests (edge cases: k = 1, k = n, duplicate points, NaN/inf rejection)

**Done when:** KMeans works on dense and sparse data and its contracts verify in CI.

## Phase 4 — PCA (`tpt-ml-proven-decomposition`)
- [ ] Crate scaffold depending only on core + needed tpt-math leaves
- [ ] Integrate tpt-math-linalg-dense eigensolver / SVD
- [ ] Centering, component selection, explained variance (+ ratio)
- [ ] `fit` / `transform` / `inverse_transform` / `fit_transform`
- [ ] Deterministic sign convention for components
- [ ] Generic over `f32` and `f64`
- [ ] `docs/contracts/pca.tptc`: requires (rows ≥ 2, 1 ≤ n_components ≤ min(rows, cols)); ensures (orthogonality, unit norm, Σ explained_variance ≤ total_variance)
- [ ] Runtime assertions with numerical tolerance for orthogonality/unit-norm
- [ ] Unit + property tests (rank-deficient data, constant columns, n_components bounds)

**Done when:** PCA contracts verify in CI and results are stable across platforms.

## Phase 5 — scikit-learn parity & benchmarks
- [ ] Pinned generator script (offline, uses a fixed scikit-learn version) producing KMeans and PCA reference vectors
- [ ] Commit fixtures (so CI never needs scikit-learn for correctness)
- [ ] Parity tests in `f64` within 1e-9 tolerance (KMeans, PCA)
- [ ] Document unavoidable differences (init RNG, sign flips) and how tests normalise them
- [ ] Benchmark harness: KMeans.fit on 1M × 50 `f32`
- [ ] Comparison script against scikit-learn on same hardware (target ≥ 2× faster)
- [ ] CI benchmark job as non-blocking report; record results in `docs/`
- [ ] Profile and optimise if below 2× target

**Done when:** parity tests pass at 1e-9 and benchmark results are published.

## Phase 6 — FFI surface (`tpt-ml-proven-ffi`, `ffi/`)
- [ ] Generate bindings through the tpt-telos FFI bridge (generated, not hand-written)
- [ ] **Python:** PyO3 + maturin, abi3 wheels for Python 3.9+; scikit-learn-style API (`KMeans(n_clusters=8).fit(X).predict(X)`)
- [ ] Python: zero-copy NumPy buffer handling; release GIL during fit
- [ ] Python: wheels for Linux, macOS, Windows installable without a C compiler (test in clean containers/VMs)
- [ ] **Go:** cgo package at `ffi/go` (module `github.com/tpt-solutions/tpt-ml-proven/ffi/go`), Go 1.21+, idiomatic API (`NewKMeans`, `Fit`, `Predict`)
- [ ] Go: prebuilt static libs / headers so builds need no cgo gymnastics
- [ ] **WASM:** wasm-bindgen / wasm-pack build exposing KMeans and PCA
- [ ] Binding tests per language (Python, Go, WASM) including f32/f64 and error propagation
- [ ] Usage examples for each language
- [ ] CI jobs building and testing all three bindings

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
Each item gets: crate code, `.tptc` contract, scikit-learn parity fixtures, FFI exposure, tests.

### Clustering
- [ ] DBSCAN
- [ ] HDBSCAN

### Decomposition
- [ ] TruncatedSVD (sparse-friendly)
- [ ] NMF

### Linear models (`tpt-ml-proven-linear`)
- [ ] Ridge
- [ ] Lasso
- [ ] ElasticNet
- [ ] Logistic regression

### Preprocessing (`tpt-ml-proven-preprocessing`)
- [ ] Scalers (Standard, MinMax, Robust)
- [ ] Encoders (OneHot, Ordinal)
- [ ] Imputers (Simple)

### Ensemble
- [ ] Stump forests (numeric verification story)

- [ ] Update README/docs and release as v0.x increments

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
