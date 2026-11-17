// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Toy reference estimators that exercise every core trait.
//!
//! These are not production algorithms. They exist to prove the traits compose
//! and to give the contract macros a real `fit` to sit in. They are also a
//! worked example for implementing the traits in the algorithm crates.
//!
//! * [`ColumnCenterer`]: an unsupervised estimator and transformer.
//! * [`MeanRegressor`]: a supervised estimator and predictor.

use alloc::vec::Vec;

use crate::error::{Error, FitError};
use crate::float::Float;
use crate::traits::{
    feature_mismatch, Predictor, SupervisedEstimator, Transformer, UnsupervisedEstimator,
};
use crate::view::{ArrayView, Dense, TargetView};
use crate::{ensures, requires};

/// Subtracts each column's mean. Fitting records the means.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ColumnCenterer<F: Float> {
    means: Option<Vec<F>>,
}

impl<F: Float> ColumnCenterer<F> {
    /// Creates an unfitted centerer.
    pub fn new() -> Self {
        Self { means: None }
    }

    /// The fitted per-column means, or `None` before `fit`.
    pub fn means(&self) -> Option<&[F]> {
        self.means.as_deref()
    }
}

impl<F: Float> UnsupervisedEstimator for ColumnCenterer<F> {
    type Float = F;
    type FitError = FitError;

    fn fit<X>(&mut self, x: X) -> Result<(), FitError>
    where
        X: ArrayView<Elem = F>,
    {
        let (rows, cols) = (x.rows(), x.cols());
        requires!(rows >= 1, "column_centerer.rows_positive");
        requires!(cols >= 1, "column_centerer.cols_positive");
        if !x.all_finite() {
            return Err(Error::NonFinite { what: "X" });
        }

        let mut sums = alloc::vec![F::ZERO; cols];
        let mut mins = alloc::vec![F::ZERO; cols];
        let mut maxs = alloc::vec![F::ZERO; cols];
        for c in 0..cols {
            mins[c] = x.get(0, c);
            maxs[c] = x.get(0, c);
        }
        for r in 0..rows {
            for c in 0..cols {
                let v = x.get(r, c);
                sums[c] = sums[c] + v;
                if v < mins[c] {
                    mins[c] = v;
                }
                if v > maxs[c] {
                    maxs[c] = v;
                }
            }
        }

        let n = F::from_usize(rows);
        let means: Vec<F> = sums.iter().map(|&s| s / n).collect();

        // Postcondition: each mean lies in the column's [min, max] range, i.e.
        // in the convex hull of the column's values.
        for c in 0..cols {
            ensures!(
                means[c] >= mins[c] && means[c] <= maxs[c],
                "column_centerer.mean_in_range"
            );
        }

        self.means = Some(means);
        Ok(())
    }
}

impl<F: Float> Transformer for ColumnCenterer<F> {
    type Float = F;
    type Output = Dense<F>;

    fn transform<X>(&self, x: X) -> Result<Dense<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        let means = self.means.as_deref().ok_or(Error::NotFitted)?;
        if x.cols() != means.len() {
            return Err(feature_mismatch(x.rows(), x.cols()));
        }
        let mut out = Vec::with_capacity(x.rows() * x.cols());
        for r in 0..x.rows() {
            for (c, &m) in means.iter().enumerate() {
                out.push(x.get(r, c) - m);
            }
        }
        Ok(Dense::from_vec(out, x.rows(), x.cols())?)
    }

    fn fit_transform<X>(&mut self, x: X) -> Result<Dense<F>, FitError>
    where
        X: ArrayView<Elem = F>,
    {
        // Fit on `x`, then transform the same `x`. `&X` is an `ArrayView`, so
        // the input is read twice without being copied.
        UnsupervisedEstimator::fit(self, &x)?;
        self.transform(x)
    }
}

/// Predicts the mean of the training targets, whatever the features.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeanRegressor<F: Float> {
    mean: Option<F>,
    n_features: usize,
}

impl<F: Float> MeanRegressor<F> {
    /// Creates an unfitted regressor.
    pub fn new() -> Self {
        Self {
            mean: None,
            n_features: 0,
        }
    }

    /// The fitted target mean, or `None` before `fit`.
    pub fn mean(&self) -> Option<F> {
        self.mean
    }
}

impl<F: Float> SupervisedEstimator for MeanRegressor<F> {
    type Float = F;
    type FitError = FitError;

    fn fit<X, Y>(&mut self, x: X, y: Y) -> Result<(), FitError>
    where
        X: ArrayView<Elem = F>,
        Y: TargetView<Elem = F>,
    {
        requires!(x.rows() >= 1, "mean_regressor.rows_positive");
        requires!(x.rows() == y.len(), "mean_regressor.rows_match_targets");
        if !x.all_finite() {
            return Err(Error::NonFinite { what: "X" });
        }
        let mut sum = F::ZERO;
        for i in 0..y.len() {
            let t = y.get(i);
            if !t.is_finite() {
                return Err(Error::NonFinite { what: "y" });
            }
            sum = sum + t;
        }
        self.mean = Some(sum / F::from_usize(y.len()));
        self.n_features = x.cols();
        Ok(())
    }
}

impl<F: Float> Predictor for MeanRegressor<F> {
    type Float = F;
    type Output = Vec<F>;

    fn predict<X>(&self, x: X) -> Result<Vec<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        let mean = self.mean.ok_or(Error::NotFitted)?;
        if x.cols() != self.n_features {
            return Err(feature_mismatch(x.rows(), x.cols()));
        }
        Ok(alloc::vec![mean; x.rows()])
    }
}
