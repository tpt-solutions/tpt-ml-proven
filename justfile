# Developer recipes for tpt-ml-proven. Install `just` (https://github.com/casey/just) to run them.
# Each recipe is what CI runs, so `just ci` reproduces the main pipeline locally.

set shell := ["bash", "-cu"]

default:
    @just --list

# Everything the main CI pipeline checks, in order.
ci: fmt clippy test header contracts deny

# Formatting, as CI checks it.
fmt:
    cargo fmt --all --check

# Lints, warnings as errors.
clippy:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

# Rust tests for the whole workspace.
test:
    cargo test --workspace --all-features

# Licence and advisory policy (deny.toml).
deny:
    cargo deny check

# Every tracked Rust file starts with the SPDX header.
header:
    #!/usr/bin/env bash
    set -eu
    missing=$(git ls-files '*.rs' ':!:tools/tptc-verify/tests/fixtures/**' | while read -r f; do
        grep -q "SPDX-License-Identifier: MIT OR Apache-2.0" "$f" || echo "$f"
    done)
    if [ -n "$missing" ]; then
        echo "files without the SPDX header:"; echo "$missing"; exit 1
    fi
    echo "all Rust files carry the SPDX header"

# Solver-verify every contract in docs/contracts/ (fails on a disproved or unlinked clause).
contracts:
    cargo run --manifest-path tools/tptc-verify/Cargo.toml -- docs/contracts

# Python binding: build a wheel, install it, run pytest.
py-test:
    cd ffi/python && python -m maturin build --release -o "${TMPDIR:-/tmp}/tpt-ml-wheels"
    python -m pip install --force-reinstall "${TMPDIR:-/tmp}"/tpt-ml-wheels/*.whl
    cd ffi/python && python -m pytest -q

# WebAssembly binding: Node tests through wasm-pack.
wasm-test:
    cd ffi/wasm && wasm-pack test --node

# Go binding: build the C library first, then test the cgo package.
go-test:
    cargo build --release --manifest-path ffi/c/Cargo.toml
    cd ffi/go && go test ./...

# Benchmark KMeans against scikit-learn (report only; see docs/benchmarks.md).
bench:
    python tools/parity/bench_kmeans.py
    cargo run --release -p tpt-ml-proven-clustering --example bench_kmeans --features parallel -- target/bench/kmeans_1000000x50_f32.bin 1000000 50 8 20 1,$(nproc)
