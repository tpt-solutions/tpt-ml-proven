// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Estimator traits, modelled on scikit-learn's method names with Rust types.
//!
//! Differences from scikit-learn:
//!
//! * `fit` returns `Result`. It never fails silently or leaves a half-fitted
//!   model.
//! * `predict` and `transform` return `Result` too, so a shape mismatch or an
//!   unfitted model is an error value, not a panic.
//! * Inputs are generic over [`ArrayView`] / [`TargetView`], so borrowed data
//!   is used without a copy.
//! * Composition is plain generic code (see the crate docs), not a runtime
//!   pipeline object.
//!
//! The input type is a method-level generic rather than an associated type. A
//! borrowed view carries a lifetime, and an associated type cannot express "any
//! lifetime" without generic associated types. The element type is an associated
//! type, which keeps `f32` / `f64` choice in the estimator.

use crate::error::{Error, FitError, ShapeError};
use crate::float::Float;
use crate::view::{ArrayView, TargetView};

/// An estimator that learns from `(X, y)` pairs.
pub trait SupervisedEstimator {
    /// Element type of `X` and `y`.
    type Float: Float;
    /// Error returned by `fit`.
    type FitError: core::fmt::Debug;

    /// Learns from features `x` and targets `y`.
    fn fit<X, Y>(&mut self, x: X, y: Y) -> Result<(), Self::FitError>
    where
        X: ArrayView<Elem = Self::Float>,
        Y: TargetView<Elem = Self::Float>;
}

/// An estimator that learns from `X` alone.
pub trait UnsupervisedEstimator {
    /// Element type of `X`.
    type Float: Float;
    /// Error returned by `fit`.
    type FitError: core::fmt::Debug;

    /// Learns from features `x`.
    fn fit<X>(&mut self, x: X) -> Result<(), Self::FitError>
    where
        X: ArrayView<Elem = Self::Float>;
}

/// Anything that can produce predictions from a fitted model.
pub trait Predictor {
    /// Element type of `X`.
    type Float: Float;
    /// Prediction output (for example a label vector or a matrix).
    type Output;

    /// Predicts for each row of `x`.
    ///
    /// Returns [`Error::NotFitted`] before `fit`, and [`Error::Shape`] if `x`
    /// does not match the fitted feature count.
    fn predict<X>(&self, x: X) -> Result<Self::Output, Error>
    where
        X: ArrayView<Elem = Self::Float>;
}

/// Anything that transforms data: preprocessing or decomposition.
pub trait Transformer {
    /// Element type of `X`.
    type Float: Float;
    /// Transformed output.
    type Output;

    /// Transforms `x` with the fitted state.
    ///
    /// Returns [`Error::NotFitted`] before `fit`, and [`Error::Shape`] on a
    /// feature-count mismatch.
    fn transform<X>(&self, x: X) -> Result<Self::Output, Error>
    where
        X: ArrayView<Elem = Self::Float>;

    /// Fits to `x`, then transforms it.
    fn fit_transform<X>(&mut self, x: X) -> Result<Self::Output, FitError>
    where
        X: ArrayView<Elem = Self::Float>;
}

/// Convenience: a shape error for a feature-count mismatch.
pub(crate) fn feature_mismatch(rows: usize, cols: usize) -> Error {
    Error::Shape(ShapeError {
        rows,
        cols,
        reason: "feature count differs from the fitted model",
    })
}
