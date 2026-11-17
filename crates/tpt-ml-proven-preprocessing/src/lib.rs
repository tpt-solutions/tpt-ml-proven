// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Preprocessing transformers for tpt-ml-proven.
//!
//! * [`StandardScaler`], [`MinMaxScaler`] and [`RobustScaler`] rescale each feature.
//! * [`OneHotEncoder`] and [`OrdinalEncoder`] encode categorical columns, given as
//!   numbers.
//! * [`SimpleImputer`] fills missing values (NaN).
//!
//! Statistics are computed in `f64`, whatever the element type, and results are rounded
//! to the element type. Conventions follow scikit-learn: population standard deviation,
//! and a zero range or spread is replaced by 1 so that the transform stays finite.

pub mod encoders;
pub mod impute;
pub mod scalers;

pub use encoders::{OneHotEncoder, OrdinalEncoder};
pub use impute::{ImputeStrategy, SimpleImputer};
pub use scalers::{MinMaxScaler, RobustScaler, StandardScaler};

/// Replaces a zero scale with 1, as scikit-learn does.
pub(crate) fn nonzero(scale: f64) -> f64 {
    if scale == 0.0 {
        1.0
    } else {
        scale
    }
}
