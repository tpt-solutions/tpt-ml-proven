// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//go:build !windows

package mlproven

// Build the static library first: cargo build --release --manifest-path ../c/Cargo.toml

/*
#cgo LDFLAGS: -L${SRCDIR}/../c/target/release -ltpt_ml_proven -ldl -lpthread -lm
*/
import "C"
