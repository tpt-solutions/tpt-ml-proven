// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Matrix decomposition estimators for tpt-ml-proven.
//!
//! [`Pca`] is principal component analysis, with the contract in
//! `docs/contracts/pca.tptc`. Its eigendecomposition is a Jacobi solver written
//! here, so this crate has no dependencies beyond core.

pub mod eigen;
pub mod pca;
pub mod svd;

pub use pca::Pca;
pub use svd::TruncatedSvd;
