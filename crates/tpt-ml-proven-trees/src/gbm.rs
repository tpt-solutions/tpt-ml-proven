// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! The public estimators: a single regression tree, and gradient boosting over many trees.

use tpt_ml_proven_core::error::{Error, ShapeError};
use tpt_ml_proven_core::{
    ensures, requires, ArrayView, Predictor, SupervisedEstimator, TargetView,
};

use crate::tree::{fit_tree, Limits, Tree};

/// Reads a feature matrix and target into flat `f64` buffers, checking shapes and values.
fn read_xy<X, Y>(x: &X, y: &Y) -> Result<(Vec<f64>, Vec<f64>, usize, usize), Error>
where
    X: ArrayView<Elem = f64>,
    Y: TargetView<Elem = f64>,
{
    let (rows, cols) = (x.rows(), x.cols());
    if rows == 0 || cols == 0 {
        return Err(Error::EmptyInput);
    }
    if y.len() != rows {
        return Err(Error::Shape(ShapeError {
            rows,
            cols,
            reason: "target length differs from the number of rows",
        }));
    }
    if !x.all_finite() {
        return Err(Error::NonFinite { what: "X" });
    }
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            data.push(x.get(r, c));
        }
    }
    let targets: Vec<f64> = (0..rows).map(|i| y.get(i)).collect();
    if targets.iter().any(|v| !v.is_finite()) {
        return Err(Error::NonFinite { what: "y" });
    }
    Ok((data, targets, rows, cols))
}

/// Rows of a feature matrix as `f64`, with the column count checked against a fitted model.
fn read_rows<X: ArrayView<Elem = f64>>(x: &X, n_features: usize) -> Result<Vec<f64>, Error> {
    if x.cols() != n_features {
        return Err(Error::Shape(ShapeError {
            rows: x.rows(),
            cols: x.cols(),
            reason: "feature count differs from the fitted model",
        }));
    }
    if !x.all_finite() {
        return Err(Error::NonFinite { what: "X" });
    }
    let mut out = Vec::with_capacity(x.rows() * x.cols());
    for r in 0..x.rows() {
        for c in 0..x.cols() {
            out.push(x.get(r, c));
        }
    }
    Ok(out)
}

fn check_limits(max_depth: usize, min_samples_leaf: usize) -> Result<(), Error> {
    if max_depth == 0 {
        return Err(Error::InvalidParameter {
            name: "max_depth",
            reason: "must be at least 1",
        });
    }
    if min_samples_leaf == 0 {
        return Err(Error::InvalidParameter {
            name: "min_samples_leaf",
            reason: "must be at least 1",
        });
    }
    Ok(())
}

// ---------------------------------------------------------------- single tree

/// A single CART regression tree.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionTreeRegressor {
    max_depth: usize,
    min_samples_leaf: usize,
    fitted: Option<Tree>,
}

impl DecisionTreeRegressor {
    /// An unfitted tree of at most `max_depth` levels, with at least one row per leaf.
    pub fn new(max_depth: usize) -> Self {
        Self {
            max_depth,
            min_samples_leaf: 1,
            fitted: None,
        }
    }

    /// Sets the smallest number of training rows a leaf may hold (default 1).
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n;
        self
    }

    /// The fitted tree, or `None` before `fit`.
    pub fn tree(&self) -> Option<&Tree> {
        self.fitted.as_ref()
    }
}

impl DecisionTreeRegressor {
    pub(crate) fn max_depth(&self) -> usize {
        self.max_depth
    }

    pub(crate) fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    pub(crate) fn from_parts(
        max_depth: usize,
        min_samples_leaf: usize,
        fitted: Option<Tree>,
    ) -> Self {
        Self {
            max_depth,
            min_samples_leaf,
            fitted,
        }
    }
}

impl SupervisedEstimator for DecisionTreeRegressor {
    type Float = f64;
    type FitError = Error;

    fn fit<X, Y>(&mut self, x: X, y: Y) -> Result<(), Error>
    where
        X: ArrayView<Elem = f64>,
        Y: TargetView<Elem = f64>,
    {
        self.fitted = None;
        check_limits(self.max_depth, self.min_samples_leaf)?;
        let (data, targets, rows, cols) = read_xy(&x, &y)?;
        let tree = fit_tree(
            &data,
            rows,
            cols,
            &targets,
            Limits {
                max_depth: self.max_depth,
                min_samples_leaf: self.min_samples_leaf,
            },
        )?;
        self.fitted = Some(tree);
        Ok(())
    }
}

impl Predictor for DecisionTreeRegressor {
    type Float = f64;
    type Output = Vec<f64>;

    fn predict<X>(&self, x: X) -> Result<Vec<f64>, Error>
    where
        X: ArrayView<Elem = f64>,
    {
        let tree = self.fitted.as_ref().ok_or(Error::NotFitted)?;
        let data = read_rows(&x, tree.n_features)?;
        Ok(data
            .chunks(tree.n_features)
            .map(|row| tree.predict_row(row))
            .collect())
    }
}

// ---------------------------------------------------------------- gradient boosting

/// Gradient boosting with squared-error loss and regression trees as the base learners.
#[derive(Debug, Clone, PartialEq)]
pub struct GradientBoostingRegressor {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_leaf: usize,
    fitted: Option<Boosted>,
}

/// The fitted state: the starting prediction and the trees, in the order they were added.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Boosted {
    pub(crate) init: f64,
    pub(crate) trees: Vec<Tree>,
    pub(crate) n_features: usize,
    pub(crate) train_loss: Vec<f64>,
    pub(crate) learning_rate: f64,
}

impl GradientBoostingRegressor {
    /// An unfitted model with `n_estimators` trees of depth `max_depth`, learning rate 0.1,
    /// and at least one row per leaf.
    pub fn new(n_estimators: usize, max_depth: usize) -> Self {
        Self {
            n_estimators,
            learning_rate: 0.1,
            max_depth,
            min_samples_leaf: 1,
            fitted: None,
        }
    }

    /// Sets the shrinkage applied to each tree, in `(0, 1]` (default 0.1).
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the smallest number of training rows a leaf may hold (default 1).
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n;
        self
    }

    /// Number of fitted trees, or `None` before `fit`.
    pub fn n_estimators_fitted(&self) -> Option<usize> {
        self.fitted.as_ref().map(|b| b.trees.len())
    }

    /// Mean squared training error after each round, or `None` before `fit`. Never increases.
    pub fn train_loss(&self) -> Option<&[f64]> {
        self.fitted.as_ref().map(|b| b.train_loss.as_slice())
    }

    /// The fitted trees, or `None` before `fit`.
    pub fn trees(&self) -> Option<&[Tree]> {
        self.fitted.as_ref().map(|b| b.trees.as_slice())
    }
}

impl GradientBoostingRegressor {
    pub(crate) fn n_estimators(&self) -> usize {
        self.n_estimators
    }

    pub(crate) fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    pub(crate) fn max_depth(&self) -> usize {
        self.max_depth
    }

    pub(crate) fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    pub(crate) fn boosted(&self) -> Option<&Boosted> {
        self.fitted.as_ref()
    }

    pub(crate) fn from_parts(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        fitted: Option<Boosted>,
    ) -> Self {
        Self {
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            fitted,
        }
    }
}

impl SupervisedEstimator for GradientBoostingRegressor {
    type Float = f64;
    type FitError = Error;

    fn fit<X, Y>(&mut self, x: X, y: Y) -> Result<(), Error>
    where
        X: ArrayView<Elem = f64>,
        Y: TargetView<Elem = f64>,
    {
        self.fitted = None;
        if self.n_estimators == 0 {
            return Err(Error::InvalidParameter {
                name: "n_estimators",
                reason: "must be at least 1",
            });
        }
        if !(self.learning_rate.is_finite()
            && self.learning_rate > 0.0
            && self.learning_rate <= 1.0)
        {
            return Err(Error::InvalidParameter {
                name: "learning_rate",
                reason: "must be in (0, 1]",
            });
        }
        check_limits(self.max_depth, self.min_samples_leaf)?;
        let (data, targets, rows, cols) = read_xy(&x, &y)?;
        requires!(rows >= 1, "gbm.rows_positive");
        requires!(self.n_estimators >= 1, "gbm.estimators_positive");
        requires!(
            self.learning_rate > 0.0 && self.learning_rate <= 1.0,
            "gbm.learning_rate_unit"
        );

        let n = rows as f64;
        let init = targets.iter().sum::<f64>() / n;
        let mut pred = vec![init; rows];
        let limits = Limits {
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
        };
        let mut trees = Vec::with_capacity(self.n_estimators);
        let mut train_loss = Vec::with_capacity(self.n_estimators);
        let mut residual = vec![0.0; rows];
        for _ in 0..self.n_estimators {
            for i in 0..rows {
                residual[i] = targets[i] - pred[i];
            }
            let before = residual.iter().map(|r| r * r).sum::<f64>() / n;
            let tree = fit_tree(&data, rows, cols, &residual, limits)?;
            for i in 0..rows {
                pred[i] += self.learning_rate * tree.predict_row(&data[i * cols..(i + 1) * cols]);
            }
            let after = targets
                .iter()
                .zip(&pred)
                .map(|(t, p)| (t - p) * (t - p))
                .sum::<f64>()
                / n;
            // A round never makes the training error worse (see the crate docs).
            ensures!(
                after <= before * (1.0 + 1e-12) + 1e-300,
                "gbm.loss_non_increasing"
            );
            train_loss.push(after);
            trees.push(tree);
        }
        self.fitted = Some(Boosted {
            init,
            trees,
            n_features: cols,
            train_loss,
            learning_rate: self.learning_rate,
        });
        Ok(())
    }
}

impl Predictor for GradientBoostingRegressor {
    type Float = f64;
    type Output = Vec<f64>;

    fn predict<X>(&self, x: X) -> Result<Vec<f64>, Error>
    where
        X: ArrayView<Elem = f64>,
    {
        let model = self.fitted.as_ref().ok_or(Error::NotFitted)?;
        let data = read_rows(&x, model.n_features)?;
        let cols = model.n_features;
        Ok(data
            .chunks(cols)
            .map(|row| {
                // Same order as training: start from the mean, add each shrunken tree.
                let mut p = model.init;
                for tree in &model.trees {
                    p += model.learning_rate * tree.predict_row(row);
                }
                p
            })
            .collect())
    }
}

// ---------------------------------------------------------------- score

impl DecisionTreeRegressor {
    /// Coefficient of determination R² of the predictions for `x` against `y`.
    pub fn score<X: ArrayView<Elem = f64>>(&self, x: X, y: &[f64]) -> Result<f64, Error> {
        let pred = Predictor::predict(self, x)?;
        tpt_ml_proven_core::metrics::r2(y, &pred)
    }
}

// ---------------------------------------------------------------- score

impl GradientBoostingRegressor {
    /// Coefficient of determination R² of the predictions for `x` against `y`.
    pub fn score<X: ArrayView<Elem = f64>>(&self, x: X, y: &[f64]) -> Result<f64, Error> {
        let pred = Predictor::predict(self, x)?;
        tpt_ml_proven_core::metrics::r2(y, &pred)
    }
}
