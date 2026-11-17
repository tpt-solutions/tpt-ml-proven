# tpt-ml-proven

Classical machine learning in Rust with **proven correctness**: core algorithms carry
`requires` / `ensures` contracts checked by a self-contained QF_LRA solver.

> Status: pre-alpha. Phase 0 (project setup) only. No algorithms are implemented yet.

## Goals

- **Proven correctness** — contracts, not just tests.
- **Modern API** — type-safe estimators, zero-copy where possible, `no_std` + `alloc` core.
- **Scikit-compatible surface** — familiar `fit` / `transform` / `predict` semantics.
- **FFI-first** — Python (PyO3) and Go (cgo) bindings, generated rather than hand-rolled.

See [docs/DESIGN.md](docs/DESIGN.md) for the full design specification.

## Workspace

| Crate | Purpose |
| --- | --- |
| `tpt-ml-proven-core` | Estimator traits, contracts, shared errors (`no_std` + `alloc`) |
| `tpt-ml-proven-clustering` | KMeans, DBSCAN, HDBSCAN |
| `tpt-ml-proven-decomposition` | PCA, TruncatedSVD, NMF |
| `tpt-ml-proven-linear` | Ridge, Lasso, ElasticNet, Logistic regression |
| `tpt-ml-proven-preprocessing` | Scalers, encoders, imputers |
| `tpt-ml-proven-ffi` | PyO3 and cgo glue |

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
