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
- [ ] Profile and optimise if below 2× target — done: single-thread 6.26 s → 1.61 s with the fused pass. Currently 1.16× slower single-threaded, and level with scikit-learn on all cores (0.42 s against 0.43 s). The 2× target is not met (see `docs/benchmarks.md`)

**Done when:** parity tests pass at 1e-9 and benchmark results are published.

## Phase 6 — FFI surface (`tpt-ml-proven-ffi`, `ffi/`)
- [ ] ~~Generate bindings through the tpt-telos FFI bridge~~ — superseded: the bridge rejects float, array and generic types at the FFI boundary (only integers are supported), so bindings are generated per language by standard tools (PyO3/maturin, wasm-bindgen, cbindgen for cgo). Decision recorded in `docs/bindings.md`
- [x] **Python:** PyO3 + maturin, abi3 wheels for Python 3.9+ (built as `cp39-abi3`); scikit-learn-style API, with `fit` returning self (`KMeans(n_clusters=8).fit(X).predict(X)`)
- [x] Python: zero-copy NumPy input via `__array_interface__` (the buffer protocol is not in the stable ABI); GIL released during `fit`
- [ ] Python: wheels for Linux, macOS, Windows installable without a C compiler (test in clean containers/VMs). Windows build works locally; Linux/macOS and clean-VM installs not yet tested
- [x] **Go:** cgo package at `ffi/go` (module `github.com/tpt-solutions/tpt-ml-proven/ffi/go`), Go 1.21+, idiomatic API (`NewKMeans`, `Fit`, `Predict`). KMeans and PCA over the C ABI, with options and `Close`. Phase 8 estimators not yet in Go
- [ ] Go: prebuilt static libs / headers so builds need no cgo gymnastics
- [x] **WASM:** wasm-bindgen / wasm-pack build exposing KMeans and PCA (`ffi/wasm`, Node target)
- [x] Binding tests for Python (12 for KMeans and PCA, plus 11 for the Phase 8 estimators), WASM (7 plus 7 for Phase 8), and Go (6), including f32/f64 and error propagation
- [x] Usage examples for Python, WASM and Go (`ffi/python/README.md`, `ffi/wasm/README.md`, `ffi/go/README.md`)
- [ ] CI jobs building and testing all three bindings. Added: Python wheel on Linux, macOS and Windows (pytest); WASM (Node tests); C ABI (tests and header check); Go (`go-bindings` job, Linux). Not yet run on GitHub. Locally, Go tests pass on Windows, linking the GNU build of the C library through the MSYS2 MinGW gcc

**Done when:** KMeans and PCA run from Python, Go and WASM in CI on all three OSes.

## Phase 7 — v0.1 release
- [x] README with one-paragraph verification story
- [x] API docs (rustdoc, builds warning-free with `-D warnings`), usage guide (`docs/usage.md`), contract docs (`docs/contracts/README.md`)
- [x] CHANGELOG (root and per crate, updated for Phases 8 to 10)
- [ ] Success-criteria sign-off (spec §10):
  - [ ] KMeans and PCA match scikit-learn vectors within 1e-9
  - [ ] All `docs/contracts/` solver-verified in CI
  - [ ] Python wheel installs on Linux/macOS/Windows with no C compiler
  - [ ] Go package builds without cgo header gymnastics
  - [ ] KMeans 1M × 50 f32 ≥ 2× faster than scikit-learn
  - [ ] README explains verification in one paragraph
- [ ] Final cargo-deny license audit (no Apache-only deps) — `cargo deny check` passes locally on the root workspace (advisories, bans, licenses); the bindings are separate workspaces with their own `deny.toml`. Re-run before release
- [ ] Publish crates to crates.io (core → algorithm crates → ffi)
- [ ] Publish wheels to PyPI (trusted publishing)
- [ ] Tag `v0.1.0` and create GitHub release (Go module picks up the tag)

**Done when:** v0.1.0 is installable from crates.io, PyPI and via `go get`.

## Phase 8 — Roadmap: remaining algorithms
Each item gets: crate code, `.tptc` contract, scikit-learn parity fixtures, FFI exposure, tests. Done so far: crate code, parity fixtures and tests, and `.tptc` contracts with runtime checks. FFI: Python, WASM and C expose these. Go covers KMeans and PCA only.

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

- [x] Update README/docs and release as v0.x increments — every crate README and CHANGELOG, and the root README and CHANGELOG, are updated for these. Releasing is still open (Phase 7)

## Phase 9 — v0.2
- [x] Design stable on-disk model format (replace interim bincode/serde). Specified in `docs/model-format.md`, implemented in `tpt-ml-proven-core::persist`. Chose our own dependency-free format over serde/bincode: see the reasons in the spec
- [x] Interim: serde/bincode support for models behind a feature flag (v0.1.x). Superseded: not implemented, because the dependency-free format covers it. Recorded in `docs/model-format.md`
- [x] Format versioning and backward-compatibility tests. Golden version 1 files for Ridge, KMeans and PCA are committed and loaded on every test run. Newer versions are refused
- [ ] Go vanity import path `tptsolutions.co.nz/mlproven` (go-import meta tags, module path migration). Meta-tag page written in `docs/site/go/mlproven/`, but not live. The `go.mod` migration is a breaking change, and waits for the domain to serve the page
- [x] Documentation site: a static landing page in `docs/site/`, with links to the docs. It is hand-written, with no generator

## Phase 10 — Future work
- [x] `tpt-ml-proven-trees` crate: CART regression and gradient boosting, with structural invariants as contracts (`decision-tree.tptc`, `gradient-boosting.tptc`) and scikit-learn parity. Random forests, classification trees and subsampling are not yet implemented
- [ ] GPU backends (after tpt-gpu is stable)
- [ ] npm publishing for the WASM package (optional)

---

## Phase 11 — Hardening and stale-doc fixes
- [x] Verify the unchecked basics first: `no_std` core builds without default features, `cargo deny check`, parity tests (all pass locally; `cargo deny` has one harmless unmatched-Zlib warning)
- [x] StumpForest `predict`: store `n_features` and reject any other width (it accepts wider input silently)
- [x] HDBSCAN: replace `.expect("finite distances")` with `total_cmp` (`hdbscan.rs`, 4 sites)
- [x] Fix the `trees/lib.rs` doc reference to the missing `trees.tptc`
- [x] Own `.tptc` and runtime checks for Lasso
- [x] `.tptc` contracts for the preprocessing crate (scalers, encoders, imputer)
- [x] Model format: dtype tag in the header (v2 writes, v1 still loads), reject NaN/inf on load, `Writer` support for `Vec<Vec<f64>>`
- [x] Persistence for TruncatedSvd, Nmf, StumpForest, the three scalers, both encoders and SimpleImputer, with golden files
- [x] Make `TruncatedSvd::components()` and `Nmf::components()` match PCA's `&Dense<F>`; add SVD `inverse_transform` and `explained_variance`
- [x] Refresh stale docs: benchmark figures in this file, contradictions in `docs/benchmarks.md`, the Go section of `docs/bindings.md`
- [x] Crate metadata for publishing: keywords, categories, per-crate `readme`, `rust-version`, `[package.metadata.docs.rs]`; decide the fate of the empty `tpt-ml-proven-ffi` shell (marked `publish = false`; `rust-version = 1.85`, verified)

## Phase 12 — Core usability (Rust)
- [x] `core::metrics`: r2, mse/rmse/mae, accuracy, log_loss, silhouette, adjusted_rand; `score()` on regressors and classifiers (KMeans is unsupervised and has none; the tree models have `score` too)
- [x] `core::model_selection`: seeded `train_test_split`, `KFold`, `cross_val_score`, plus a row-selection helper (`select_rows`, `select_values`)
- [ ] Static, zero-cost `Pipeline`/`then` helper (keeps "no runtime Pipeline object"); `fit_predict` and `inverse_transform` trait defaults. Done: `Pipeline` and `Then`. Still open: the two trait defaults
- [x] `Default` for estimators with sensible defaults (ridge, lasso, elastic net, both logistic models, StumpForest, imputer, plus the scalers and encoders that already had one; not the trees or KMeans, which have no obvious default)
- [x] Umbrella crate `tpt-ml-proven` with a `prelude` and feature-gated re-exports

## Phase 13 — Sparse-aware algorithms
- [x] `ArrayView::for_each_nonzero` and `CsrView::row`; `all_finite` over non-zeros only
- [x] TruncatedSvd Gram and transform from non-zeros (the explained variance uses them too)
- [ ] KMeans sparse distances via precomputed centre norms; no full densify when parallel. Open: the fused, parallel kernel is shared with Phase 14, and a new distance formula would move the 1e-9 parity results, so it needs its own measured change
- [ ] Done: Ridge, ElasticNet and Lasso with implicit centring, from the non-zeros only (the `ArrayView::is_sparse` path). Lasso's own optimality check still copies the matrix. Done: NMF sparse `X·Hᵀ` and `WᵀX`, and the error from the non-zeros; DBSCAN and HDBSCAN read sparse rows and measure distances over their non-zeros
- [ ] Property tests: sparse and dense results agree for every touched estimator. Done for TruncatedSvd and NMF (`decomposition/tests/sparse.rs`), for Ridge, ElasticNet and Lasso (`linear/tests/sparse.rs`), and for DBSCAN and HDBSCAN (`clustering/tests/sparse.rs`). Open for KMeans only, whose sparse distance needs its own measured change (see the item above)

## Phase 14 — KMeans speed
- [ ] Register-blocked kernel (one row against 4 centres, one horizontal reduce per centre)
- [ ] Argmin on `‖c‖²−2x·c`, centre norms per iteration, inertia computed directly (keep 1e-9 parity)
- [ ] Move `lo`/`hi` hull tracking out of the hot loop
- [ ] `u32` labels, in-pass change count, reused per-chunk buffers
- [ ] Re-measure and record in `docs/benchmarks.md`, including if the 2× target is still missed (Hamerly and runtime AVX2 already rejected; do not retry)

## Phase 15 — Trees: classification, forests, subsampling
- [ ] Make `trees` generic over `Float`
- [x] `DecisionTreeClassifier` (gini, entropy). Parity against scikit-learn on both criteria; kind 19 with a golden file; contract `decision-tree-classifier.tptc`
- [x] `RandomForestRegressor` and `RandomForestClassifier` (bootstrap, `max_features`, seeded, kinds 20 and 21 with golden files). Not done: the parallel feature. The forests are tested by properties, not scikit-learn fixtures
- [ ] GBM `subsample` and `max_features`; `min_samples_split`, `max_leaf_nodes`, feature importances. Done: `subsample`, `max_features` and the seed (kind 22). Done for the single trees, forests and boosting: `min_samples_split`, `max_leaf_nodes` (best-first growth) and `feature_importances` (kinds 26 to 30). The C ABI exposes the same settings and importances. Open: the boosting importances' small gap to scikit-learn (see the trees changelog)
- [ ] Presort per feature once instead of re-sorting per node
- [ ] `.tptc` contracts, runtime checks, persistence kinds and golden files, scikit-learn parity fixtures

## Phase 16 — Python ergonomics
- [x] `.pyi` stubs, `py.typed`, mixed-layout `python-source` (the compiled module is `tpt_ml_proven._native`; stubs cover the 11 estimators the binding exposes)
- [x] `get_params`/`set_params`, `__repr__`, `fit_predict`, `score`, `n_features_in_`. Done: `get_params`, `set_params` and `__repr__` on every estimator; `score` on the supervised estimators except DBSCAN and HDBSCAN (which have none); `fit_predict` on KMeans, DBSCAN and HDBSCAN; `n_features_in_` on every estimator that has a fitted width (absent before fit)
- [x] Pickle via `__reduce__` and `save`/`load` over the model file format. Done for KMeans, PCA, TruncatedSVD, NMF, Ridge, Lasso, ElasticNet and StumpForest; loading rebuilds the settings from the file. Open: LogisticRegression (its binary wrapper keeps a class-label map the file format has no place for) and DBSCAN/HDBSCAN (no file format yet)
- [x] NumPy array returns; pandas/polars input; copy non-contiguous and cast int input instead of raising. Done: array results are NumPy arrays when NumPy is installed; pandas and polars frames, lists of rows and pandas Series targets are accepted. Non-contiguous and integer arrays are converted, not rejected. Pickle and the file format are unaffected
- [x] Expose scalers, encoders, imputer, tree models and new classifiers; scikit-learn `Pipeline` interop test. Done: StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, SimpleImputer, DecisionTreeRegressor, DecisionTreeClassifier, GradientBoostingRegressor, RandomForestRegressor and RandomForestClassifier, each with the full estimator surface; the interop tests (a scikit-learn scaler in front of our estimators, clone, cross-validation, grid search, KMeans in a pipeline) and scikit-learn comparisons where the algorithm is exact. Not exposed: the Rust-side ones still missing from Python (none of the planned estimators remain)
- [x] Fix `pyproject.toml` classifiers and URLs and the stale Cargo description; example notebook. Done: descriptions and the typed classifier; the example notebook is `ffi/python/examples/tour.ipynb`, executed end to end (clustering, save and load, a scikit-learn pipeline, trees and forests, a plot)

## Phase 17 — CLI and verify command
- [x] `tpt-ml-proven-cli` crate, binary `tpt-ml`: `fit`, `predict`, `transform` from CSV, `info <model>` (not published)
- [x] `tpt-ml verify <model> [--data x.csv]`: check magic, version, checksum and invariants, then re-run runtime contract checks on the data (the data check runs predict or transform, which executes the runtime checks; it does not re-run each contract clause on its own)
- [ ] No new dependencies (hand-rolled CSV and argument parsing); integration tests using the golden model files, including a corrupted-byte failure case. Done: no new dependencies, and the corrupted-byte test. Still open: the tests fit fresh models and do not load the golden files

## Phase 18 — Go, WASM and C completeness
- [x] C ABI: expose the Phase 11 to 15 estimators; `tpml_version()`; pkg-config file; `examples/c/main.c` compiled in CI
- [x] Go: wrap DBSCAN, HDBSCAN, SVD, NMF, Ridge, Lasso, ElasticNet, Logistic, StumpForest; `PCA.InverseTransform`, KMeans `n_iter`, `float32`
- [ ] Go: document the `ffi/go/vX.Y.Z` tag and prebuilt static lib story; macOS and Windows link files checked
- [x] WASM: options-object constructors (no positional `undefined`, no BigInt seeds); expose transformers and trees
- [ ] WASM: templated `package.json`, `web` and `bundler` targets, browser demo

## Phase 19 — Tooling, governance, examples, docs
- [x] `justfile`: `ci`, `test`, `py-test`, `wasm-test`, `go-test`, `header`, `contracts`, `bench`. `just` is not installed on this machine, so the recipes were not run through it. The `header` script was run by hand, and the other steps are the commands CI already runs
- [ ] `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`, issue and PR templates, `.devcontainer/`; update `CONTRIBUTING.md` (still says issues only). Done: the first four, and `.devcontainer/`. Still open: `CONTRIBUTING.md`, whose accepts-issues-only policy is the maintainers' call to change
- [x] `.github/dependabot.yml`: cargo for each of the five lockfile directories, github-actions, pip
- [ ] `examples/`: Rust CSV to KMeans to save and load, regression with `cross_val_score`, Python notebook, wasm demo, C `main.c`; all built in CI. Done: the two Rust examples (`save_and_load`, `cross_validate` in `crates/tpt-ml-proven/examples/`, built by `cargo test`); the Python notebook (`ffi/python/examples/tour.ipynb`, committed with executed outputs); `main.c` (compiled and run in CI). Still open: the wasm demo has not been run in a browser, and the notebook is not built in CI
- [ ] Hosted docs (mdBook or extended `docs/site`): per-language quickstart, "which estimator", scikit-learn migration guide, parameter tables, contract explainer; GitHub Pages workflow
- [x] Root README: install section and a 5-line quickstart

## Deferred (blocked on GitHub push, accounts or domain)
- [ ] `release.yml`: tag-triggered wheels (manylinux, musllinux, macOS universal2, Windows, sdist), PyPI trusted publishing, crates.io in dependency order, C ABI tarballs, `ffi/go/vX.Y.Z` tags
- [ ] `cargo publish --dry-run` job in CI
- [ ] Correct the Go tag in Phase 7: a subdirectory module needs `ffi/go/vX.Y.Z`, not `vX.Y.Z`
