# tpt-ml-proven

Classical machine learning in Rust with **proven correctness**: core algorithms carry
`requires` / `ensures` contracts checked by a self-contained QF_LRA solver.

> Status: pre-release (v0.1 in progress). KMeans and PCA are implemented, verified
> against scikit-learn, and available from Rust, Python and WebAssembly. The Phase 8
> algorithms (DBSCAN, HDBSCAN, TruncatedSVD, NMF, the linear models and a stump forest) are in
> Rust with contracts and are exposed through the Python and WebAssembly bindings. The C ABI
> covers all of them. The preprocessing transformers are not yet exposed in any
> binding. The Go binding is still to come.

## Verification

Each algorithm has a contract in `docs/contracts/`, written as `requires` and `ensures`
clauses. The linear clauses are checked by a QF_LRA solver, `tpt-solver`, and every
UNSAT answer is accepted only after its certificate is checked. Clauses outside linear
arithmetic (for example monotone inertia descent or orthogonality) are checked at run
time by runtime assertions. CI fails if an implemented algorithm has a contract clause
without its runtime check. Results are compared
with scikit-learn to 1e-9, on committed fixtures. The verifier and the solver are
separate, dev-only tools; the library crates have no solver dependency.

## Bindings

- Python: `ffi/python`, PyO3 and maturin, `abi3` wheels for Python 3.9 and later.
- WebAssembly: `ffi/wasm`, wasm-bindgen.
- C ABI: `ffi/c`, with a generated header, for the planned Go binding.

See `docs/bindings.md` for the decisions and what is still open.

## Goals

- **Proven correctness** — contracts, not just tests.
- **Modern API** — type-safe estimators, zero-copy where possible, `no_std` + `alloc` core.
- **Scikit-compatible surface** — familiar `fit` / `transform` / `predict` semantics.
- **FFI-first** — Python (PyO3) and Go (cgo) bindings, generated rather than hand-rolled.

See [docs/DESIGN.md](docs/DESIGN.md) for the full design specification.

## Workspace

Each crate has its own README and changelog.

| Crate | Status | Purpose | Docs |
| --- | --- | --- | --- |
| `tpt-ml-proven-core` | Implemented | Estimator traits, contracts, shared errors (`no_std` + `alloc`) | [README](crates/tpt-ml-proven-core/README.md) · [CHANGELOG](crates/tpt-ml-proven-core/CHANGELOG.md) |
| `tpt-ml-proven-clustering` | KMeans, DBSCAN, HDBSCAN implemented | Clustering estimators | [README](crates/tpt-ml-proven-clustering/README.md) · [CHANGELOG](crates/tpt-ml-proven-clustering/CHANGELOG.md) |
| `tpt-ml-proven-decomposition` | PCA, TruncatedSVD, NMF implemented | Matrix decomposition estimators | [README](crates/tpt-ml-proven-decomposition/README.md) · [CHANGELOG](crates/tpt-ml-proven-decomposition/CHANGELOG.md) |
| `tpt-ml-proven-linear` | Ridge, Lasso, ElasticNet, logistic regression (binary and multinomial) implemented | [README](crates/tpt-ml-proven-linear/README.md) · [CHANGELOG](crates/tpt-ml-proven-linear/CHANGELOG.md) |
| `tpt-ml-proven-preprocessing` | Scalers, encoders, imputers implemented | [README](crates/tpt-ml-proven-preprocessing/README.md) · [CHANGELOG](crates/tpt-ml-proven-preprocessing/CHANGELOG.md) |
| `tpt-ml-proven-ensemble` | Stump forest implemented | Bagged decision stumps | [README](crates/tpt-ml-proven-ensemble/README.md) · [CHANGELOG](crates/tpt-ml-proven-ensemble/CHANGELOG.md) |
| `tpt-ml-proven-ffi` | Placeholder | FFI surface; the bindings live in `ffi/` | [README](crates/tpt-ml-proven-ffi/README.md) · [CHANGELOG](crates/tpt-ml-proven-ffi/CHANGELOG.md) |

Workspace-wide changes (contracts, the verifier, parity tests, bindings) are recorded in
the [root changelog](CHANGELOG.md).

## Building

```sh
cargo build --workspace
cargo test --workspace
```

The toolchain is pinned to `stable` via [rust-toolchain.toml](rust-toolchain.toml).

## Licence

Dual-licensed under either of

- [MIT licence](LICENSE-MIT)
- [Apache licence, version 2.0](LICENSE-APACHE)

at your option.

Copyright (c) 2026 TPT Solutions.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
