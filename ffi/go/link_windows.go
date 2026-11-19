// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//go:build windows

package mlproven

// The Rust static library must be built for the GNU target, because cgo links with the
// MinGW toolchain: cargo build --release --target x86_64-pc-windows-gnu --manifest-path ../c/Cargo.toml

/*
// Link the static archive by path: the same target also builds a DLL import library, which
// the linker would otherwise pick.
#cgo LDFLAGS: ${SRCDIR}/../c/target/x86_64-pc-windows-gnu/release/libtpt_ml_proven.a -lws2_32 -luserenv -lbcrypt -lntdll -lkernel32 -ladvapi32 -lsynchronization
*/
import "C"
