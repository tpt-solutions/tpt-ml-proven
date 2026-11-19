// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//go:build !windows

package mlproven

// Build the static library first: cargo build --release --manifest-path ../c/Cargo.toml

/*
// Link the static archive by path. The crate also builds a shared library, which `-ltpt_ml_proven`
// would pick; it lives in the build directory, not on the loader's search path, so the test
// binary would fail to start with "cannot open shared object file".
#cgo LDFLAGS: ${SRCDIR}/../c/target/release/libtpt_ml_proven.a -ldl -lpthread -lm
*/
import "C"
