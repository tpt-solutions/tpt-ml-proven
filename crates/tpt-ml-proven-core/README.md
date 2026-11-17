# tpt-ml-proven-core

Estimator traits, contract glue and shared errors for
[tpt-ml-proven](https://github.com/tpt-solutions/tpt-ml-proven).

This crate is `no_std` + `alloc` and has no dependencies. Every other crate in the
workspace builds on it.

> Status: implemented, pre-release.

## What it provides

- **Traits**: `SupervisedEstimator` and `UnsupervisedEstimator` (`fit`), `Predictor`
  (`predict`), and `Transformer` (`transform`, `fit_transform`).
- **Input views**: `ArrayView` and `TargetView` abstract over the input layout.
  `DenseView` borrows row-major data without copying, `Dense` owns it, and `CsrView`
  holds sparse rows.
- **Numeric type**: the `Float` trait, implemented for `f32` and `f64`.
- **Errors**: `Error`, `FitError`, `ShapeError` and `ContractViolation`.
- **Contract macros**: `requires!`, `ensures!` and `static_require!`, which tie code
  to the clauses in `docs/contracts/`.
- **Toy estimators** in `toy`, `ColumnCenterer` and `MeanRegressor`, that exercise the
  traits end to end.

## Features

| Feature | Default | Effect |
| --- | --- | --- |
| `std` | yes | Reserved for std-only conveniences. |
| `runtime-contracts` | yes | Compiles in the data-dependent `requires!` / `ensures!` checks. Disable for hot loops. |

The crate keeps building without default features for bare-metal and WebAssembly
targets, for example `thumbv6m-none-eabi` and `wasm32-unknown-unknown`.

## Documentation

The crate documentation has the design notes on composition (generic code in place of
a runtime pipeline object) and doc-tested examples. Run `cargo doc -p tpt-ml-proven-core --open`.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Licence

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache-2.0](../../LICENSE-APACHE), at
your option.
