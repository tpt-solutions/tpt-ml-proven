# tpt-ml-proven

Classical machine learning in Rust with **proven correctness**: the algorithms carry
`requires` / `ensures` contracts, checked by a self-contained QF_LRA solver where the clauses
are linear and by runtime checks everywhere else.

> Status: pre-release (v0.1 in progress). Most of the planned algorithms are implemented and
> checked against scikit-learn on committed fixtures. The tree models, the model file format
> and the bindings are in place. Release steps (publishing, tagging, the Go import path) are
> not done.

## Quickstart

Nothing is published to a registry yet, so install from a checkout of this repository.

Rust, in a crate that depends on `tpt-ml-proven` by path:

```rust
use tpt_ml_proven::prelude::*;

let data = [0.0, 0.0, 0.1, 0.0, 10.0, 10.0, 10.1, 10.0];
let x = DenseView::new(&data, 4, 2).unwrap();
let mut km = KMeans::<f64>::new(2).with_seed(1);
UnsupervisedEstimator::fit(&mut km, x).unwrap();
let labels = Predictor::predict(&km, DenseView::new(&data, 4, 2).unwrap()).unwrap();
```

Python, after `maturin develop --release` in `ffi/python`:

```python
from tpt_ml_proven import KMeans

labels = KMeans(2, seed=1).fit(X).labels_   # X is a 2-D NumPy array
```

JavaScript, after `node scripts/build.mjs` in `ffi/wasm`:

```js
const km = new KMeansModel({ nClusters: 2, seed: 1 });
km.fit(new Float64Array([0, 0, 0.1, 0, 10, 10, 10.1, 10]), 2);
const labels = km.labels;
```

## Algorithms

| Crate | Estimators |
| --- | --- |
| [`tpt-ml-proven-clustering`](crates/tpt-ml-proven-clustering/README.md) | KMeans, DBSCAN, HDBSCAN |
| [`tpt-ml-proven-decomposition`](crates/tpt-ml-proven-decomposition/README.md) | PCA, TruncatedSVD, NMF |
| [`tpt-ml-proven-linear`](crates/tpt-ml-proven-linear/README.md) | Ridge, Lasso, ElasticNet, logistic regression (binary and multinomial) |
| [`tpt-ml-proven-preprocessing`](crates/tpt-ml-proven-preprocessing/README.md) | StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, SimpleImputer |
| [`tpt-ml-proven-ensemble`](crates/tpt-ml-proven-ensemble/README.md) | StumpForest |
| [`tpt-ml-proven-trees`](crates/tpt-ml-proven-trees/README.md) | DecisionTreeRegressor, DecisionTreeClassifier, GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier |

Each estimator's contract is in [docs/contracts/](docs/contracts/README.md). Parity with
scikit-learn is recorded in [docs/parity.md](docs/parity.md).

## Verification

Each algorithm has a contract in `docs/contracts/`, written as `requires` and `ensures`
clauses. The linear clauses are checked by a QF_LRA solver, `tpt-solver`, and every UNSAT
answer is accepted only after its certificate is checked. Clauses outside linear arithmetic
(for example monotone inertia descent, or orthogonality) are checked at run time by runtime
assertions. CI fails if an implemented algorithm has a contract clause without its runtime
check. Results are compared with scikit-learn to 1e-9 or 1e-6, depending on the algorithm, on
committed fixtures. The verifier and the solver are separate, dev-only tools; the library
crates have no solver dependency.

Tree models add invariants that are easy to state and check: the leaves of a tree partition
the training rows, depth is bounded, and each boosting round does not increase the training
error.

## Model files

Fitted models can be written to and read from a versioned binary format, specified in
[docs/model-format.md](docs/model-format.md). The format has a magic header, a version, a model
kind, little-endian fields and a checksum. Readers refuse newer versions and any file whose
contents don't fit the model's invariants. Golden files from version 1 are committed and
tested on every run.

The Rust crates, the Python package and the WebAssembly package read and write model files.
The C and Go bindings do not expose them yet.

## Bindings

| Directory | Binding | What it covers |
| --- | --- | --- |
| [`ffi/python`](ffi/python/README.md) | Python (PyO3, abi3 wheels) | Every estimator and transformer, with model files |
| [`ffi/wasm`](ffi/wasm/README.md) | WebAssembly (wasm-bindgen) | Every estimator and transformer, with model files; single precision for KMeans and PCA |
| [`ffi/c`](ffi/c) | C ABI (cbindgen header) | Every estimator and transformer; single precision for KMeans and PCA |
| [`ffi/go`](ffi/go/README.md) | Go (cgo over the C ABI) | Clustering, decomposition, the linear models and StumpForest; single precision for KMeans and PCA. Trees and transformers are not wrapped yet |

See [docs/usage.md](docs/usage.md) for examples of each binding.

## Documentation

- [Usage guide](docs/usage.md), for each binding.
- [Model file format](docs/model-format.md).
- [Contracts and the verifier](docs/contracts/README.md).
- [Design specification](docs/DESIGN.md).
- [Benchmarks against scikit-learn](docs/benchmarks.md), and [parity](docs/parity.md).
- A static landing page is in [docs/site/](docs/site/index.html).

## Goals

- **Proven correctness** — contracts, not just tests.
- **Modern API** — type-safe estimators, zero-copy where possible, `no_std` + `alloc` core.
- **Scikit-compatible surface** — familiar `fit` / `transform` / `predict` semantics.
- **FFI-first** — Python, WebAssembly, C and Go bindings, generated or thin rather than
  hand-rolled.

See [docs/DESIGN.md](docs/DESIGN.md) for the full design specification.

## Workspace

Each crate has its own README and changelog.

| Crate | Status | Purpose | Docs |
| --- | --- | --- | --- |
| `tpt-ml-proven-core` | Implemented | Estimator traits, contracts, shared errors, model file format (`no_std` + `alloc`) | [README](crates/tpt-ml-proven-core/README.md) · [CHANGELOG](crates/tpt-ml-proven-core/CHANGELOG.md) |
| `tpt-ml-proven-clustering` | KMeans, DBSCAN, HDBSCAN implemented | Clustering estimators | [README](crates/tpt-ml-proven-clustering/README.md) · [CHANGELOG](crates/tpt-ml-proven-clustering/CHANGELOG.md) |
| `tpt-ml-proven-decomposition` | PCA, TruncatedSVD, NMF implemented | Matrix decomposition estimators | [README](crates/tpt-ml-proven-decomposition/README.md) · [CHANGELOG](crates/tpt-ml-proven-decomposition/CHANGELOG.md) |
| `tpt-ml-proven-linear` | Ridge, Lasso, ElasticNet, logistic implemented | Linear models | [README](crates/tpt-ml-proven-linear/README.md) · [CHANGELOG](crates/tpt-ml-proven-linear/CHANGELOG.md) |
| `tpt-ml-proven-preprocessing` | Scalers, encoders, imputers implemented | Preprocessing transformers | [README](crates/tpt-ml-proven-preprocessing/README.md) · [CHANGELOG](crates/tpt-ml-proven-preprocessing/CHANGELOG.md) |
| `tpt-ml-proven-ensemble` | StumpForest implemented | Bagged decision stumps | [README](crates/tpt-ml-proven-ensemble/README.md) · [CHANGELOG](crates/tpt-ml-proven-ensemble/CHANGELOG.md) |
| `tpt-ml-proven-trees` | DecisionTreeRegressor, GradientBoostingRegressor implemented | Tree models with structural invariants | [README](crates/tpt-ml-proven-trees/README.md) · [CHANGELOG](crates/tpt-ml-proven-trees/CHANGELOG.md) |
| `tpt-ml-proven-ffi` | No code of its own | Bindings live in `ffi/` | [README](crates/tpt-ml-proven-ffi/README.md) · [CHANGELOG](crates/tpt-ml-proven-ffi/CHANGELOG.md) |

Workspace-wide changes (contracts, the verifier, parity tests, bindings, the file format) are
recorded in the [root changelog](CHANGELOG.md).

## Building

```sh
cargo build --workspace
cargo test --workspace
```

The toolchain is pinned to `stable` via [rust-toolchain.toml](rust-toolchain.toml). The
bindings build from their own directories; see each binding's README.

## Licence

Dual-licensed under either of

- [MIT licence](LICENSE-MIT)
- [Apache licence, version 2.0](LICENSE-APACHE)

at your option.

Copyright (c) 2026 TPT Solutions.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
