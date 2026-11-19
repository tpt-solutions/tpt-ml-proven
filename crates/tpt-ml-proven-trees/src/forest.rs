// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Random forests: many CART trees, each grown on a bootstrap sample of the rows and choosing
//! every split from a random subset of the features.
//!
//! Randomness comes from one seeded [`SplitMix64`], so a fixed seed gives the same forest on
//! every platform. The sampling differs from scikit-learn's, so forests are not compared with
//! it value by value. They are tested by properties instead: a separable problem is learned,
//! the same seed gives the same forest, and predictions stay inside the training labels.
//!
//! Defaults follow scikit-learn: a regressor considers every feature at each split, and a
//! classifier considers `⌊√n_features⌋`. Regression averages the trees' predictions.
//! Classification takes a majority vote over the trees' predicted labels, with ties going to
//! the smaller label. scikit-learn averages class probabilities instead, so the two can differ
//! on close votes.

use tpt_ml_proven_core::error::{Error, ShapeError};
use tpt_ml_proven_core::rng::SplitMix64;
use tpt_ml_proven_core::{
    ensures, requires, ArrayView, Predictor, SupervisedEstimator, TargetView,
};

use crate::classifier::{grow_classification_tree, Criterion};
use crate::gbm::{check_limits, read_xy};
use crate::tree::{fit_tree, Limits, Node, Tree};

/// Row indices drawn with replacement, `rows` of them.
fn bootstrap(rows: usize, rng: &mut SplitMix64) -> Vec<usize> {
    (0..rows).map(|_| rng.next_below(rows)).collect()
}

/// The rows at `idx`, copied out of the row-major `data`.
pub(crate) fn take_rows(data: &[f64], cols: usize, idx: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(idx.len() * cols);
    for &i in idx {
        out.extend_from_slice(&data[i * cols..(i + 1) * cols]);
    }
    out
}

/// Features per split: the configured count, capped at the number of features, or the default.
fn features_per_split(configured: Option<usize>, default: usize, cols: usize) -> usize {
    configured.unwrap_or(default).clamp(1, cols)
}

fn check_forest(n_estimators: usize) -> Result<(), Error> {
    if n_estimators == 0 {
        return Err(Error::InvalidParameter {
            name: "n_estimators",
            reason: "must be at least 1",
        });
    }
    Ok(())
}

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

// ---------------------------------------------------------------- regressor

/// A random forest of regression trees, predicting the mean of the trees' predictions.
#[derive(Debug, Clone, PartialEq)]
pub struct RandomForestRegressor {
    pub(crate) n_estimators: usize,
    pub(crate) max_depth: usize,
    pub(crate) min_samples_leaf: usize,
    pub(crate) max_features: Option<usize>,
    pub(crate) seed: u64,
    pub(crate) fitted: Option<(Vec<Tree>, usize)>,
}

impl RandomForestRegressor {
    /// An unfitted forest of `n_estimators` trees of at most `max_depth` levels.
    pub fn new(n_estimators: usize, max_depth: usize) -> Self {
        Self {
            n_estimators,
            max_depth,
            min_samples_leaf: 1,
            max_features: None,
            seed: 0,
            fitted: None,
        }
    }

    /// Sets the seed for the bootstrap samples and the feature choices.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets how many features each split considers.
    pub fn with_max_features(mut self, max_features: usize) -> Self {
        self.max_features = Some(max_features);
        self
    }

    /// Sets the minimum number of bootstrap rows in each leaf.
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Number of fitted trees, or `None` before `fit`.
    pub fn n_estimators(&self) -> Option<usize> {
        self.fitted.as_ref().map(|(t, _)| t.len())
    }

    /// R² of the predictions for `x` against `y`.
    pub fn score<X: ArrayView<Elem = f64>>(&self, x: X, y: &[f64]) -> Result<f64, Error> {
        let pred = Predictor::predict(self, x)?;
        tpt_ml_proven_core::metrics::r2(y, &pred)
    }
}

impl SupervisedEstimator for RandomForestRegressor {
    type Float = f64;
    type FitError = Error;

    fn fit<X, Y>(&mut self, x: X, y: Y) -> Result<(), Error>
    where
        X: ArrayView<Elem = f64>,
        Y: TargetView<Elem = f64>,
    {
        self.fitted = None;
        check_forest(self.n_estimators)?;
        check_limits(self.max_depth, self.min_samples_leaf)?;
        let (data, targets, rows, cols) = read_xy(&x, &y)?;
        let per_split = features_per_split(self.max_features, cols, cols);
        let limits = Limits {
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            max_features: per_split,
        };
        let mut rng = SplitMix64::new(self.seed);
        let mut trees = Vec::with_capacity(self.n_estimators);
        for _ in 0..self.n_estimators {
            let idx = bootstrap(rows, &mut rng);
            let boot_x = take_rows(&data, cols, &idx);
            let boot_y: Vec<f64> = idx.iter().map(|&i| targets[i]).collect();
            trees.push(fit_tree(&boot_x, rows, cols, &boot_y, limits, &mut rng)?);
        }
        ensures!(trees.len() == self.n_estimators, "forest.estimators_fitted");
        self.fitted = Some((trees, cols));
        Ok(())
    }
}

impl Predictor for RandomForestRegressor {
    type Float = f64;
    type Output = Vec<f64>;

    /// The mean over the trees of each row's predicted value.
    fn predict<X>(&self, x: X) -> Result<Vec<f64>, Error>
    where
        X: ArrayView<Elem = f64>,
    {
        let (trees, n_features) = self.fitted.as_ref().ok_or(Error::NotFitted)?;
        let flat = read_rows(&x, *n_features)?;
        let cols = *n_features;
        let out = flat
            .chunks(cols)
            .map(|row| {
                let sum: f64 = trees.iter().map(|t| t.predict_row(row)).sum();
                sum / trees.len() as f64
            })
            .collect();
        Ok(out)
    }
}

// ---------------------------------------------------------------- classifier

/// A random forest of classification trees, predicting the majority vote of the trees.
#[derive(Debug, Clone, PartialEq)]
pub struct RandomForestClassifier {
    pub(crate) n_estimators: usize,
    pub(crate) max_depth: usize,
    pub(crate) min_samples_leaf: usize,
    pub(crate) criterion: Criterion,
    pub(crate) max_features: Option<usize>,
    pub(crate) seed: u64,
    pub(crate) fitted: Option<ForestFit>,
}

/// The learned part of a classification forest.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ForestFit {
    pub(crate) trees: Vec<Tree>,
    pub(crate) classes: Vec<f64>,
    pub(crate) n_features: usize,
}

impl RandomForestClassifier {
    /// An unfitted forest of `n_estimators` trees of at most `max_depth` levels, using Gini.
    pub fn new(n_estimators: usize, max_depth: usize) -> Self {
        Self {
            n_estimators,
            max_depth,
            min_samples_leaf: 1,
            criterion: Criterion::Gini,
            max_features: None,
            seed: 0,
            fitted: None,
        }
    }

    /// Sets the seed for the bootstrap samples and the feature choices.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets how many features each split considers.
    pub fn with_max_features(mut self, max_features: usize) -> Self {
        self.max_features = Some(max_features);
        self
    }

    /// Sets the minimum number of bootstrap rows in each leaf.
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Chooses the impurity measure for every tree.
    pub fn with_criterion(mut self, criterion: Criterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Number of fitted trees, or `None` before `fit`.
    pub fn n_estimators(&self) -> Option<usize> {
        self.fitted.as_ref().map(|f| f.trees.len())
    }

    /// The distinct class labels seen in training, ascending, or `None` before `fit`.
    pub fn classes(&self) -> Option<&[f64]> {
        self.fitted.as_ref().map(|f| f.classes.as_slice())
    }

    /// Fraction of rows of `x` whose predicted label equals the label in `y`.
    pub fn score<X: ArrayView<Elem = f64>>(&self, x: X, y: &[f64]) -> Result<f64, Error> {
        let pred = Predictor::predict(self, x)?;
        tpt_ml_proven_core::metrics::accuracy(y, &pred)
    }
}

impl SupervisedEstimator for RandomForestClassifier {
    type Float = f64;
    type FitError = Error;

    fn fit<X, Y>(&mut self, x: X, y: Y) -> Result<(), Error>
    where
        X: ArrayView<Elem = f64>,
        Y: TargetView<Elem = f64>,
    {
        self.fitted = None;
        check_forest(self.n_estimators)?;
        check_limits(self.max_depth, self.min_samples_leaf)?;
        let (data, labels, rows, cols) = read_xy(&x, &y)?;
        requires!(rows >= 1, "forest.rows_positive");

        let mut classes = labels.clone();
        classes.sort_by(f64::total_cmp);
        classes.dedup();
        let class_of: Vec<usize> = labels
            .iter()
            .map(|v| {
                classes
                    .binary_search_by(|c| c.total_cmp(v))
                    .expect("every label is one of the classes")
            })
            .collect();

        // The conventional default is the square root of the feature count.
        let default = ((cols as f64).sqrt().floor() as usize).max(1);
        let per_split = features_per_split(self.max_features, default, cols);
        let limits = Limits {
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            max_features: per_split,
        };
        let mut rng = SplitMix64::new(self.seed);
        let mut trees = Vec::with_capacity(self.n_estimators);
        for _ in 0..self.n_estimators {
            let idx = bootstrap(rows, &mut rng);
            let boot_x = take_rows(&data, cols, &idx);
            let boot_class: Vec<usize> = idx.iter().map(|&i| class_of[i]).collect();
            trees.push(grow_classification_tree(
                &boot_x,
                cols,
                &boot_class,
                &classes,
                limits,
                self.criterion,
                &mut rng,
            ));
        }
        ensures!(
            trees.iter().all(|t| t.nodes.iter().all(|n| match n {
                Node::Leaf { value, .. } => classes.contains(value),
                Node::Split { .. } => true,
            })),
            "forest.predicts_training_class"
        );
        self.fitted = Some(ForestFit {
            trees,
            classes,
            n_features: cols,
        });
        Ok(())
    }
}

impl Predictor for RandomForestClassifier {
    type Float = f64;
    type Output = Vec<f64>;

    /// The label most trees vote for, ties going to the smaller label.
    fn predict<X>(&self, x: X) -> Result<Vec<f64>, Error>
    where
        X: ArrayView<Elem = f64>,
    {
        let fit = self.fitted.as_ref().ok_or(Error::NotFitted)?;
        let flat = read_rows(&x, fit.n_features)?;
        let cols = fit.n_features;
        let out = flat
            .chunks(cols)
            .map(|row| {
                let mut votes: Vec<(f64, usize)> = Vec::new();
                for t in &fit.trees {
                    let label = t.predict_row(row);
                    match votes.iter_mut().find(|(l, _)| *l == label) {
                        Some(v) => v.1 += 1,
                        None => votes.push((label, 1)),
                    }
                }
                votes
                    .iter()
                    .max_by(|a, b| a.1.cmp(&b.1).then(b.0.total_cmp(&a.0)))
                    .map(|(l, _)| *l)
                    .unwrap_or(fit.classes[0])
            })
            .collect();
        Ok(out)
    }
}

// ---------------------------------------------------------------- settings

impl RandomForestRegressor {
    /// Number of trees the forest was configured to grow.
    pub fn n_estimators_setting(&self) -> usize {
        self.n_estimators
    }

    /// Maximum depth of each tree.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Minimum rows per leaf of each tree.
    pub fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    /// Features considered per split, or `None` for the default.
    pub fn max_features(&self) -> Option<usize> {
        self.max_features
    }

    /// The seed the forest was configured with.
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

impl RandomForestClassifier {
    /// Number of trees the forest was configured to grow.
    pub fn n_estimators_setting(&self) -> usize {
        self.n_estimators
    }

    /// Maximum depth of each tree.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Minimum rows per leaf of each tree.
    pub fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    /// Impurity measure of each tree.
    pub fn criterion(&self) -> Criterion {
        self.criterion
    }

    /// Features considered per split, or `None` for the default.
    pub fn max_features(&self) -> Option<usize> {
        self.max_features
    }

    /// The seed the forest was configured with.
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ml_proven_core::DenseView;

    /// Class 0 below 0.5 on the first feature and class 1 above, with a noisy second feature.
    fn separable() -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..60)
            .flat_map(|i| [i as f64 / 60.0, ((i * 37) % 11) as f64])
            .collect();
        let y: Vec<f64> = (0..60).map(|i| if i < 30 { 0.0 } else { 1.0 }).collect();
        (x, y)
    }

    #[test]
    fn a_forest_learns_a_separable_problem() {
        let (x, y) = separable();
        let mut f = RandomForestClassifier::new(25, 4).with_seed(3);
        SupervisedEstimator::fit(&mut f, DenseView::new(&x, 60, 2).unwrap(), &y[..]).unwrap();
        let acc = f.score(DenseView::new(&x, 60, 2).unwrap(), &y).unwrap();
        assert!(acc >= 0.95, "accuracy {acc}");
    }

    #[test]
    fn the_same_seed_gives_the_same_forest() {
        let (x, y) = separable();
        let fit = |seed| {
            let mut f = RandomForestClassifier::new(7, 3).with_seed(seed);
            SupervisedEstimator::fit(&mut f, DenseView::new(&x, 60, 2).unwrap(), &y[..]).unwrap();
            f
        };
        assert_eq!(fit(5), fit(5));
        assert_ne!(fit(5), fit(6));
    }

    #[test]
    fn predictions_are_training_labels() {
        let (x, y) = separable();
        let mut f = RandomForestClassifier::new(9, 3)
            .with_seed(1)
            .with_max_features(1);
        SupervisedEstimator::fit(&mut f, DenseView::new(&x, 60, 2).unwrap(), &y[..]).unwrap();
        let pred = Predictor::predict(&f, DenseView::new(&x, 60, 2).unwrap()).unwrap();
        assert!(pred.iter().all(|p| f.classes().unwrap().contains(p)));
    }

    #[test]
    fn a_regressor_fits_a_linear_trend() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
        let mut f = RandomForestRegressor::new(40, 6).with_seed(9);
        SupervisedEstimator::fit(&mut f, DenseView::new(&x, 50, 1).unwrap(), &y[..]).unwrap();
        let r2 = f.score(DenseView::new(&x, 50, 1).unwrap(), &y).unwrap();
        assert!(r2 > 0.95, "r2 {r2}");
        assert_eq!(f.n_estimators(), Some(40));
    }

    #[test]
    fn unfitted_forest_refuses_to_predict() {
        let x = [0.0_f64, 1.0];
        let f = RandomForestRegressor::new(3, 2);
        assert_eq!(
            Predictor::predict(&f, DenseView::new(&x, 2, 1).unwrap()).unwrap_err(),
            Error::NotFitted
        );
    }
}
