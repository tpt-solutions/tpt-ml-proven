# Language bindings

## Decision: per-language generators, not tpt-telos

The todo originally asked for bindings generated through the tpt-telos FFI bridge.
That bridge cannot do the job. Its parameter check (`validate_ffi_params` in
`tpt-telos-codegen/src/ffi.rs`) accepts only integer scalars and structs of
integers. It rejects floats, arrays, slices and generics. Every estimator here takes
a floating-point matrix, so none of the API can cross that boundary.

The bindings are therefore generated per language by the tools each ecosystem uses:

| Target | Tool | Status |
| --- | --- | --- |
| Python | PyO3 0.30 + maturin, `abi3-py39` | Built and tested locally (12 tests) |
| WebAssembly | wasm-bindgen 0.2 + wasm-pack, Node target | Built and tested locally (7 tests) |
| Go | cgo over a C ABI, header from cbindgen | Blocked: see below |

Extending tpt-telos to pass float buffers would make the bridge usable here. That
work sits in the tpt-telos repository and has not been started.

## Python

- `ffi/python` is a standalone crate, excluded from the runtime workspace, so the
  Python toolchain does not affect the library crates.
- Inputs are read in place from NumPy arrays through `__array_interface__`. The
  buffer protocol is not part of the stable ABI before Python 3.11, and wheels are
  built `abi3`, so it cannot be used here.
- `fit` releases the GIL. `fit` returns the estimator, so calls can be chained.

## WebAssembly

- `ffi/wasm` is standalone. Data crosses as flat, row-major typed arrays.
- `u64` seeds are BigInts in JavaScript.

## Go: blocked

The cgo package needs a C compiler. This machine has none, and `go env CGO_ENABLED`
is `0`. So the Go binding can be neither compiled nor tested here. Options:

1. Install a GCC-compatible compiler (for example MinGW-w64) on this machine, then
   build and test locally.
2. Verify in CI only, on a Linux runner, which has `gcc`.

The C ABI itself (an `ffi/c` crate built as `cdylib`/`staticlib`, with the header from
cbindgen) can be built and tested from Rust before any Go code exists. It has not been
written yet.
