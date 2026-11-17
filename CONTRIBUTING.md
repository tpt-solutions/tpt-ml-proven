# Contributing to tpt-ml-proven

Thanks for your interest. At this stage the project accepts **issues only**:
bug reports, questions and design feedback are welcome. Pull requests are not
being accepted yet, so please open an issue rather than a PR.

## Reporting issues

- Describe what you ran, what you expected, and what happened.
- Include the Rust toolchain (`rustc --version`) and platform.
- For numerical problems, include a minimal dataset and the expected result,
  and say which scikit-learn version (if any) you compared against.

## Project conventions

These apply to any contribution once contributions are open.

### Licensing

- The project is dual-licensed `MIT OR Apache-2.0`.
- Every Rust source file starts with the SPDX header:

  ```rust
  // SPDX-License-Identifier: MIT OR Apache-2.0
  // Copyright (c) 2026 TPT Solutions
  ```

- **Runtime dependencies** must be MIT, MIT/Apache-2.0 dual, BSD, ISC, Zlib,
  Unlicense or CC0. Apache-2.0-only dependencies are not allowed. `cargo deny check`
  enforces this via [deny.toml](deny.toml).
- Build and development tools may use any OSS licence.

### Code quality

- `cargo fmt --all --check` must pass (see [rustfmt.toml](rustfmt.toml)).
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` must pass.
- `cargo test --workspace --all-features` must pass on Linux, macOS and Windows.
- `tpt-ml-proven-core` must keep building with `--no-default-features` for a
  bare-metal target (`thumbv6m-none-eabi`) and for `wasm32-unknown-unknown`.

### Algorithms

Each algorithm ships with its `.tptc` contract in `docs/contracts/`, scikit-learn
parity fixtures, FFI exposure and tests. See [docs/DESIGN.md](docs/DESIGN.md) and
the roadmap in `todo.md`.
