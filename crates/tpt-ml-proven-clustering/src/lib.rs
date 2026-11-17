// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Clustering estimators for tpt-ml-proven.
//!
//! [`KMeans`] implements Lloyd's algorithm with k-means++ or random
//! initialisation. Its contract is `docs/contracts/kmeans.tptc`.
//!
//! Enable the `parallel` feature for a multi-threaded assignment step. The
//! results are the same for any thread count.

pub mod dbscan;
pub mod kmeans;
mod rng;

pub use dbscan::Dbscan;
pub use kmeans::{Init, KMeans};
