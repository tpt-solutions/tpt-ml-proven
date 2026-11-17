// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Estimator traits, contract glue and shared errors for tpt-ml-proven.
//!
//! This crate is `no_std` and only depends on `alloc`. The `std` feature
//! (enabled by default) is reserved for std-only conveniences. The
//! `runtime-contracts` feature (enabled by default) turns on the data-dependent
//! `requires!` / `ensures!` checks.
//!
//! # Traits
//!
//! * [`SupervisedEstimator`] and [`UnsupervisedEstimator`]: `fit` returns `Result`.
//! * [`Predictor`]: `predict` after `fit`.
//! * [`Transformer`]: `transform` and `fit_transform`.
//!
//! Inputs are [`ArrayView`] and [`TargetView`], so borrowed buffers are used
//! without copying. Element types are generic over [`Float`] (`f32` or `f64`).
//!
//! # Composition
//!
//! There is no runtime pipeline object. Steps are chained with ordinary generic
//! code, and each step's output is the next step's input:
//!
//! ```
//! use tpt_ml_proven_core::toy::{ColumnCenterer, MeanRegressor};
//! use tpt_ml_proven_core::view::DenseView;
//! use tpt_ml_proven_core::traits::{Predictor, SupervisedEstimator, Transformer};
//! use tpt_ml_proven_core::Error;
//!
//! fn fit_then_predict(raw: &[f64], rows: usize, cols: usize, y: &[f64]) -> Result<Vec<f64>, Error> {
//!     let x = DenseView::new(raw, rows, cols)?;
//!     // Step 1: a transformer. Its output is an owned matrix.
//!     let mut centerer = ColumnCenterer::<f64>::new();
//!     let centred = centerer.fit_transform(x)?;
//!     // Step 2: a supervised estimator reads the transformed matrix.
//!     let mut model = MeanRegressor::<f64>::new();
//!     model.fit(centred.view(), y)?;
//!     // Predict on new data, passed through the same fitted transformer.
//!     let new = centerer.transform(DenseView::new(raw, rows, cols)?)?;
//!     model.predict(new.view())
//! }
//!
//! let raw = [1.0, 2.0, 3.0, 4.0];
//! let y = [10.0, 20.0];
//! assert_eq!(fit_then_predict(&raw, 2, 2, &y).unwrap(), vec![15.0, 15.0]);
//! ```
//!
//! # Contracts
//!
//! [`requires!`] and [`ensures!`] check data-dependent clauses at runtime.
//! [`static_require!`] checks constant clauses at compile time. See the
//! [`contract`] module.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod contract;
pub mod error;
pub mod float;
pub mod toy;
pub mod traits;
pub mod view;

pub use error::{ContractKind, ContractViolation, Error, FitError, ShapeError};
pub use float::Float;
pub use traits::{Predictor, SupervisedEstimator, Transformer, UnsupervisedEstimator};
pub use view::{ArrayView, CsrView, Dense, DenseView, TargetView};
