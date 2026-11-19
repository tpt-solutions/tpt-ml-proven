// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Entry point for the `tpt-ml` binary. All logic is in the library.

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let stdout = std::io::stdout();
    let stderr = std::io::stderr();
    let status = tpt_ml_proven_cli::run(&argv, &mut stdout.lock(), &mut stderr.lock());
    std::process::exit(status);
}
