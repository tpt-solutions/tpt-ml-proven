// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Ensemble models for tpt-ml-proven.
//!
//! [`StumpForest`] is a bagged ensemble of decision stumps (depth-one trees) for
//! classification. Each stump is fitted on a bootstrap sample, splits one feature at one
//! threshold to minimise weighted Gini impurity, and predicts a class on each side. The
//! forest predicts by majority vote. There is no scikit-learn reference for this model, so
//! it is tested through its properties rather than by parity.

use tpt_ml_proven_core::error::{Error, ShapeError};
use tpt_ml_proven_core::{
    ensures, requires, ArrayView, Float, Predictor, SupervisedEstimator, TargetView,
};

/// SplitMix64 step; a private copy to keep the crate dependency-free.
fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One depth-one tree: `x[feature] <= threshold` predicts `left`, otherwise `right`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Stump {
    feature: usize,
    threshold: f64,
    left: f64,
    right: f64,
}

/// Gini impurity of a class-count vector.
fn gini(counts: &[usize]) -> f64 {
    let n: usize = counts.iter().sum();
    if n == 0 {
        return 0.0;
    }
    let n = n as f64;
    1.0 - counts.iter().map(|&c| (c as f64 / n).powi(2)).sum::<f64>()
}

/// Index of the largest count; ties go to the smallest index.
fn argmax(counts: &[usize]) -> usize {
    let mut best = 0;
    for (i, &c) in counts.iter().enumerate() {
        if c > counts[best] {
            best = i;
        }
    }
    best
}

/// Fits one stump to the sample `idx` (row indices into `x`), with class indices `y`.
fn fit_stump(x: &[f64], cols: usize, y: &[usize], idx: &[usize], n_classes: usize) -> Stump {
    let mut total = vec![0usize; n_classes];
    for &i in idx {
        total[y[i]] += 1;
    }
    let majority = argmax(&total) as f64;
    // Threshold f64::MAX sends every finite value left, like an infinite threshold would.
    // It stays finite so that a saved model passes the finite-value check on load.
    let mut best = Stump {
        feature: 0,
        threshold: f64::MAX,
        left: majority,
        right: majority,
    };
    let n = idx.len() as f64;
    let mut best_impurity = gini(&total);

    for feature in 0..cols {
        let mut order: Vec<usize> = idx.to_vec();
        order.sort_by(|&a, &b| x[a * cols + feature].total_cmp(&x[b * cols + feature]));
        let mut left = vec![0usize; n_classes];
        let mut right = total.clone();
        for k in 0..order.len().saturating_sub(1) {
            let i = order[k];
            left[y[i]] += 1;
            right[y[i]] -= 1;
            let v = x[i * cols + feature];
            let next = x[order[k + 1] * cols + feature];
            if v == next {
                continue;
            }
            let nl = (k + 1) as f64;
            let impurity = nl / n * gini(&left) + (n - nl) / n * gini(&right);
            if impurity < best_impurity {
                best_impurity = impurity;
                best = Stump {
                    feature,
                    threshold: 0.5 * (v + next),
                    left: argmax(&left) as f64,
                    right: argmax(&right) as f64,
                };
            }
        }
    }
    best
}

/// Bagged decision stumps for classification.
#[derive(Debug, Clone, PartialEq)]
pub struct StumpForest<F: Float> {
    n_estimators: usize,
    seed: u64,
    fitted: Option<Vec<Stump>>,
    n_features: usize,
    marker: core::marker::PhantomData<F>,
}

impl<F: Float> StumpForest<F> {
    /// An unfitted forest of `n_estimators` stumps. The seed is 0 by default.
    pub fn new(n_estimators: usize) -> Self {
        Self {
            n_estimators,
            seed: 0,
            fitted: None,
            n_features: 0,
            marker: core::marker::PhantomData,
        }
    }

    /// Sets the seed for the bootstrap samples.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Number of fitted stumps, or `None` before `fit`.
    pub fn n_estimators(&self) -> Option<usize> {
        self.fitted.as_ref().map(|s| s.len())
    }
}

impl<F: Float> SupervisedEstimator for StumpForest<F> {
    type Float = F;
    type FitError = Error;

    /// Fits on class labels given as numbers. Each distinct label is a class.
    fn fit<X, Y>(&mut self, x: X, y: Y) -> Result<(), Error>
    where
        X: ArrayView<Elem = F>,
        Y: TargetView<Elem = F>,
    {
        self.fitted = None;
        let (rows, cols) = (x.rows(), x.cols());
        if rows == 0 || cols == 0 {
            return Err(Error::EmptyInput);
        }
        if self.n_estimators == 0 {
            return Err(Error::InvalidParameter {
                name: "n_estimators",
                reason: "must be at least 1",
            });
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
        requires!(rows >= 1, "stump_forest.rows_positive");
        requires!(self.n_estimators >= 1, "stump_forest.estimators_positive");

        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(x.get(r, c).to_f64());
            }
        }
        let labels: Vec<f64> = (0..rows).map(|i| y.get(i).to_f64()).collect();
        if labels.iter().any(|v| !v.is_finite()) {
            return Err(Error::NonFinite { what: "y" });
        }
        let mut classes: Vec<f64> = labels.clone();
        classes.sort_by(f64::total_cmp);
        classes.dedup();
        let class_index: Vec<usize> = labels
            .iter()
            .map(|v| {
                classes
                    .binary_search_by(|c| c.total_cmp(v))
                    .expect("every label is one of the classes")
            })
            .collect();

        let mut state = self.seed;
        let mut stumps = Vec::with_capacity(self.n_estimators);
        for _ in 0..self.n_estimators {
            let sample: Vec<usize> = (0..rows)
                .map(|_| (next_u64(&mut state) % rows as u64) as usize)
                .collect();
            stumps.push(fit_stump(&data, cols, &class_index, &sample, classes.len()));
        }
        // Every stump predicts one of the training classes.
        ensures!(
            stumps
                .iter()
                .all(|s| classes.contains(&s.left) && classes.contains(&s.right)),
            "stump_forest.predicts_training_class"
        );
        self.fitted = Some(stumps);
        self.n_features = cols;
        Ok(())
    }
}

impl<F: Float> Predictor for StumpForest<F> {
    type Float = F;
    type Output = Vec<F>;

    /// Majority vote of the stumps for each row. Ties go to the smaller label.
    fn predict<X>(&self, x: X) -> Result<Vec<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        let stumps = self.fitted.as_ref().ok_or(Error::NotFitted)?;
        let cols = x.cols();
        if cols != self.n_features {
            return Err(Error::Shape(ShapeError {
                rows: x.rows(),
                cols,
                reason: "feature count differs from the fitted model",
            }));
        }
        if !x.all_finite() {
            return Err(Error::NonFinite { what: "X" });
        }
        Ok((0..x.rows())
            .map(|r| {
                let mut votes: Vec<(f64, usize)> = Vec::new();
                for s in stumps {
                    let v = x.get(r, s.feature).to_f64();
                    let label = if v <= s.threshold { s.left } else { s.right };
                    match votes.iter_mut().find(|(l, _)| *l == label) {
                        Some(entry) => entry.1 += 1,
                        None => votes.push((label, 1)),
                    }
                }
                let winner = votes
                    .iter()
                    .max_by(|a, b| a.1.cmp(&b.1).then(b.0.total_cmp(&a.0)))
                    .map(|(l, _)| *l)
                    .unwrap_or(0.0);
                F::from_f64(winner)
            })
            .collect())
    }
}

// ---------------------------------------------------------------- model files

use tpt_ml_proven_core::persist::{kind, Reader, Writer};

fn invalid(reason: &'static str) -> Error {
    Error::Format { reason }
}

impl<F: Float> StumpForest<F> {
    /// Serialises the forest (settings and fitted stumps) to a model file.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut w = Writer::for_float::<F>(kind::STUMP_FOREST);
        w.usize(self.n_estimators);
        w.u64(self.seed);
        w.usize(self.n_features);
        match &self.fitted {
            None => w.bool(false),
            Some(stumps) => {
                w.bool(true);
                w.usize(stumps.len());
                for s in stumps {
                    w.usize(s.feature);
                    w.f64(s.threshold);
                    w.f64(s.left);
                    w.f64(s.right);
                }
            }
        }
        Ok(w.finish())
    }

    /// Loads a forest written by [`StumpForest::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(bytes, kind::STUMP_FOREST)?;
        let n_estimators = r.usize()?;
        let seed = r.u64()?;
        let n_features = r.usize()?;
        let fitted = if r.bool()? {
            let count = r.usize()?;
            let mut stumps = Vec::new();
            for _ in 0..count {
                stumps.push(Stump {
                    feature: r.usize()?,
                    threshold: r.f64()?,
                    left: r.f64()?,
                    right: r.f64()?,
                });
            }
            Some(stumps)
        } else {
            None
        };
        r.finish()?;
        if n_estimators == 0 {
            return Err(invalid("stump forest has no estimators"));
        }
        if let Some(stumps) = &fitted {
            if stumps.len() != n_estimators {
                return Err(invalid("stump count differs from n_estimators"));
            }
            if n_features == 0 || stumps.iter().any(|s| s.feature >= n_features) {
                return Err(invalid(
                    "stump splits on a feature outside the fitted width",
                ));
            }
        }
        Ok(Self {
            n_estimators,
            seed,
            fitted,
            n_features,
            marker: core::marker::PhantomData,
        })
    }
}

// ---------------------------------------------------------------- score

impl<F: Float> StumpForest<F> {
    /// Fraction of rows of `x` whose predicted label equals the label in `y`.
    pub fn score<X: ArrayView<Elem = F>>(&self, x: X, y: &[F]) -> Result<f64, Error> {
        let pred = Predictor::predict(self, x)?;
        tpt_ml_proven_core::metrics::accuracy(y, &pred)
    }
}

impl<F: Float> Default for StumpForest<F> {
    /// The default settings, as documented on the constructor.
    fn default() -> Self {
        Self::new(10)
    }
}

// ---------------------------------------------------------------- settings

impl<F: Float> StumpForest<F> {
    /// The number of stumps the model was configured to grow.
    pub fn n_estimators_setting(&self) -> usize {
        self.n_estimators
    }

    /// The seed the model was configured with.
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

impl<F: Float> StumpForest<F> {
    /// The number of features the fitted forest was trained on, or `None` before `fit`.
    pub fn n_features(&self) -> Option<usize> {
        self.fitted.as_ref().map(|_| self.n_features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ml_proven_core::DenseView;

    /// Two classes split on the first feature; the second feature is noise.
    fn separable() -> (Vec<f64>, Vec<f64>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..40 {
            let a = i as f64 / 10.0;
            x.extend([a, (i * 7 % 13) as f64]);
            y.push(if a < 2.0 { 0.0 } else { 1.0 });
        }
        (x, y)
    }

    #[test]
    fn separable_data_is_fitted_exactly() {
        let (x, y) = separable();
        let mut f = StumpForest::<f64>::new(25).with_seed(7);
        SupervisedEstimator::fit(&mut f, DenseView::new(&x, 40, 2).unwrap(), &y[..]).unwrap();
        let pred = Predictor::predict(&f, DenseView::new(&x, 40, 2).unwrap()).unwrap();
        assert_eq!(pred, y);
    }

    #[test]
    fn same_seed_gives_same_forest() {
        let (x, y) = separable();
        let fit = |seed| {
            let mut f = StumpForest::<f64>::new(5).with_seed(seed);
            SupervisedEstimator::fit(&mut f, DenseView::new(&x, 40, 2).unwrap(), &y[..]).unwrap();
            f
        };
        assert_eq!(fit(3), fit(3));
    }

    #[test]
    fn predicting_with_a_different_width_is_a_shape_error() {
        let (x, y) = separable();
        let mut f = StumpForest::<f64>::new(25).with_seed(7);
        SupervisedEstimator::fit(&mut f, DenseView::new(&x, 40, 2).unwrap(), &y[..]).unwrap();
        // Wider input: the stumps only read the first two columns, so this used to pass silently.
        let wide: Vec<f64> = (0..40 * 3).map(|i| i as f64).collect();
        assert!(matches!(
            Predictor::predict(&f, DenseView::new(&wide, 40, 3).unwrap()),
            Err(Error::Shape(_))
        ));
        assert!(matches!(
            Predictor::predict(&f, DenseView::new(&x[..2], 2, 1).unwrap()),
            Err(Error::Shape(_))
        ));
    }

    #[test]
    fn invalid_input_is_rejected() {
        let (x, y) = separable();
        let mut f = StumpForest::<f64>::new(0);
        assert!(matches!(
            SupervisedEstimator::fit(&mut f, DenseView::new(&x, 40, 2).unwrap(), &y[..]),
            Err(Error::InvalidParameter {
                name: "n_estimators",
                ..
            })
        ));
        let f = StumpForest::<f64>::new(3);
        assert_eq!(
            Predictor::predict(&f, DenseView::new(&x, 40, 2).unwrap()).unwrap_err(),
            Error::NotFitted
        );
    }
}

#[cfg(test)]
mod persist_tests {
    use super::*;
    use tpt_ml_proven_core::DenseView;

    fn data() -> (Vec<f64>, Vec<f64>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..40 {
            let a = i as f64 / 10.0;
            x.extend([a, (i * 7 % 13) as f64]);
            y.push(if a < 2.0 { 0.0 } else { 1.0 });
        }
        (x, y)
    }

    #[test]
    fn fitted_forest_round_trips_and_predicts_identically() {
        let (x, y) = data();
        let mut f = StumpForest::<f64>::new(15).with_seed(7);
        SupervisedEstimator::fit(&mut f, DenseView::new(&x, 40, 2).unwrap(), &y[..]).unwrap();
        let loaded = StumpForest::<f64>::from_bytes(&f.to_bytes().unwrap()).unwrap();
        assert_eq!(loaded, f);
        let a = Predictor::predict(&f, DenseView::new(&x, 40, 2).unwrap()).unwrap();
        let b = Predictor::predict(&loaded, DenseView::new(&x, 40, 2).unwrap()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn unfitted_forest_round_trips() {
        let f = StumpForest::<f64>::new(4).with_seed(2);
        assert_eq!(
            StumpForest::<f64>::from_bytes(&f.to_bytes().unwrap()).unwrap(),
            f
        );
    }

    #[test]
    fn a_stump_on_an_unknown_feature_is_refused() {
        let (x, y) = data();
        let mut f = StumpForest::<f64>::new(3);
        SupervisedEstimator::fit(&mut f, DenseView::new(&x, 40, 2).unwrap(), &y[..]).unwrap();
        // Declare a width no larger than the widest feature any stump splits on.
        let widest = f
            .fitted
            .as_ref()
            .unwrap()
            .iter()
            .map(|s| s.feature)
            .max()
            .unwrap();
        f.n_features = widest;
        let bytes = f.to_bytes().unwrap();
        assert!(matches!(
            StumpForest::<f64>::from_bytes(&bytes),
            Err(Error::Format { .. })
        ));
    }
}
