# tpt-ml-proven-ffi

FFI surface for [tpt-ml-proven](https://github.com/tpt-solutions/tpt-ml-proven).

> Status: placeholder. The crate's `lib.rs` is empty until the FFI phase. The bindings
> themselves live in the directories below.

## Where the bindings are

| Directory | Binding | Notes |
| --- | --- | --- |
| [`ffi/python`](../../ffi/python/README.md) | Python (PyO3, maturin) | `abi3` wheels for Python 3.9 and later. `KMeans` and `PCA`. |
| [`ffi/wasm`](../../ffi/wasm/README.md) | WebAssembly (wasm-bindgen) | `KMeansModel` and `PcaModel`. |
| [`ffi/c`](../../ffi/c) | C ABI (cbindgen) | Generated header in `ffi/c/include/`, for the planned Go binding. |
| [`ffi/go`](../../ffi/go/README.md) | Go (cgo) | Placeholder. |

The bindings are separate packages, not workspace members. Build them from their own
directories, as described in each README.

## Changelog

See [CHANGELOG.md](CHANGELOG.md). Changes to the bindings are recorded in the
[workspace changelog](../../CHANGELOG.md).

## Licence

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache-2.0](../../LICENSE-APACHE), at
your option.
