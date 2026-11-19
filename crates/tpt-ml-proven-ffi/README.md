# tpt-ml-proven-ffi

The foreign-function surface for [tpt-ml-proven](https://github.com/tpt-solutions/tpt-ml-proven).

> Status: the bindings live in `ffi/`, as separate packages. This crate is a workspace member
> with no code of its own, kept so the workspace layout matches the design.

## Where the bindings are

| Directory | Binding | Estimators |
| --- | --- | --- |
| [`ffi/python`](../../ffi/python/README.md) | Python (PyO3, abi3 wheels) | KMeans, PCA, and the Phase 8 estimators |
| [`ffi/wasm`](../../ffi/wasm/README.md) | WebAssembly (wasm-bindgen) | KMeans, PCA, and the Phase 8 estimators |
| [`ffi/c`](../../ffi/c) | C ABI (cbindgen header in `ffi/c/include/`) | KMeans, PCA, and all the Phase 8 estimators |
| [`ffi/go`](../../ffi/go/README.md) | Go (cgo over the C ABI) | KMeans and PCA |

None of the bindings yet expose the model files (`to_bytes` and `from_bytes`), nor the
gradient-boosting and decision-tree models in `tpt-ml-proven-trees`. Their format is in
[docs/model-format.md](../../docs/model-format.md).

The bindings are separate packages, not members of this workspace, so the Python and
WebAssembly toolchains stay out of the library builds. Build each from its own directory, as
its README describes.

## Changelog

See [CHANGELOG.md](CHANGELOG.md). Changes to the bindings are recorded in the
[workspace changelog](../../CHANGELOG.md).

## Licence

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache-2.0](../../LICENSE-APACHE), at
your option.
