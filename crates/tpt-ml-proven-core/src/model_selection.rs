// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Splitting data and estimating generalisation: seeded train/test splits, K-fold indices,
//! row selection, and cross-validation over any [`SupervisedEstimator`].
//!
//! A fixed seed gives the same split on every platform, because the shuffle uses
//! [`crate::rng::SplitMix64`].

use alloc::vec::Vec;

use crate::error::{Error, ShapeError};
use crate::rng::SplitMix64;
use crate::{ArrayView, Dense, Float, SupervisedEstimator};

/// Fisher–Yates shuffle of `0..n` with the given generator.
fn shuffled(n: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.next_below(i + 1);
        idx.swap(i, j);
    }
    idx
}

/// Splits `0..n` into a seeded training set and test set. Returns `(train, test)`, each in
/// ascending order.
///
/// `test_size` is the fraction of rows held out, in `(0, 1)`. The test set gets
/// `ceil(n · test_size)` rows, and at least one row is left for training.
pub fn train_test_split(
    n: usize,
    test_size: f64,
    seed: u64,
) -> Result<(Vec<usize>, Vec<usize>), Error> {
    if n < 2 {
        return Err(Error::InvalidParameter {
            name: "n",
            reason: "a split needs at least two rows",
        });
    }
    if !(test_size > 0.0 && test_size < 1.0) {
        return Err(Error::InvalidParameter {
            name: "test_size",
            reason: "must be a fraction strictly between 0 and 1",
        });
    }
    // Round up without `f64::ceil`, which is not available under `no_std`.
    let exact = n as f64 * test_size;
    let mut n_test = exact as usize;
    if (n_test as f64) < exact {
        n_test += 1;
    }
    let n_test = n_test.clamp(1, n - 1);
    let mut rng = SplitMix64::new(seed);
    let order = shuffled(n, &mut rng);
    let mut test = order[..n_test].to_vec();
    let mut train = order[n_test..].to_vec();
    test.sort_unstable();
    train.sort_unstable();
    Ok((train, test))
}

/// One fold: the `(train, test)` row indices.
pub type Fold = (Vec<usize>, Vec<usize>);

/// K-fold cross-validation indices.
///
/// Each row is in the test set of exactly one fold. The first `n mod k` folds have one row
/// more than the rest, as in scikit-learn. With a seed, rows are shuffled before folding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KFold {
    k: usize,
    seed: Option<u64>,
}

impl KFold {
    /// `k` folds, taken in row order.
    pub fn new(k: usize) -> Self {
        Self { k, seed: None }
    }

    /// Shuffles the rows with `seed` before folding.
    pub fn with_shuffle(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// The `(train, test)` row indices of each fold, for `n` rows.
    pub fn split(&self, n: usize) -> Result<Vec<Fold>, Error> {
        if self.k < 2 || self.k > n {
            return Err(Error::InvalidParameter {
                name: "k",
                reason: "must be at least 2 and at most the number of rows",
            });
        }
        let order = match self.seed {
            Some(seed) => shuffled(n, &mut SplitMix64::new(seed)),
            None => (0..n).collect(),
        };
        let (base, extra) = (n / self.k, n % self.k);
        let mut folds = Vec::with_capacity(self.k);
        let mut start = 0;
        for f in 0..self.k {
            let size = base + usize::from(f < extra);
            let mut test: Vec<usize> = order[start..start + size].to_vec();
            let mut train: Vec<usize> = order[..start]
                .iter()
                .chain(&order[start + size..])
                .copied()
                .collect();
            test.sort_unstable();
            train.sort_unstable();
            folds.push((train, test));
            start += size;
        }
        Ok(folds)
    }
}

/// Copies the listed rows of `x` into a new matrix, in the order given.
pub fn select_rows<F, X>(x: &X, rows: &[usize]) -> Result<Dense<F>, Error>
where
    F: Float,
    X: ArrayView<Elem = F>,
{
    let cols = x.cols();
    let mut out = Vec::with_capacity(rows.len() * cols);
    for &r in rows {
        if r >= x.rows() {
            return Err(Error::Shape(ShapeError {
                rows: x.rows(),
                cols,
                reason: "row index out of range",
            }));
        }
        for c in 0..cols {
            out.push(x.get(r, c));
        }
    }
    Ok(Dense::from_vec(out, rows.len(), cols)?)
}

/// Copies the listed entries of `y`, in the order given.
pub fn select_values<F: Float>(y: &[F], rows: &[usize]) -> Result<Vec<F>, Error> {
    rows.iter()
        .map(|&r| {
            y.get(r).copied().ok_or(Error::Shape(ShapeError {
                rows: y.len(),
                cols: 1,
                reason: "row index out of range",
            }))
        })
        .collect()
}

/// Cross-validated scores: for each fold, a fresh model from `make` is fitted on the training
/// rows and scored on the test rows by `score`. Returns one score per fold.
///
/// `score` receives the fitted model, the test rows and the test targets. For a regressor
/// that is typically `|m, x, y| metrics::r2(y, &Predictor::predict(m, x.view())?)`.
pub fn cross_val_score<F, M, Mk, S, X>(
    x: &X,
    y: &[F],
    cv: &KFold,
    mut make: Mk,
    score: S,
) -> Result<Vec<f64>, Error>
where
    F: Float,
    X: ArrayView<Elem = F>,
    M: SupervisedEstimator<Float = F, FitError = Error>,
    Mk: FnMut() -> M,
    S: Fn(&M, &Dense<F>, &[F]) -> Result<f64, Error>,
{
    if y.len() != x.rows() {
        return Err(Error::Shape(ShapeError {
            rows: x.rows(),
            cols: x.cols(),
            reason: "target length differs from the number of rows",
        }));
    }
    let mut scores = Vec::new();
    for (train, test) in cv.split(x.rows())? {
        let train_x = select_rows(x, &train)?;
        let train_y = select_values(y, &train)?;
        let test_x = select_rows(x, &test)?;
        let test_y = select_values(y, &test)?;
        let mut model = make();
        SupervisedEstimator::fit(&mut model, train_x.view(), &train_y[..])?;
        scores.push(score(&model, &test_x, &test_y)?);
    }
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DenseView;
    use alloc::vec;

    #[test]
    fn split_partitions_the_rows_and_is_seeded() {
        let (train, test) = train_test_split(10, 0.3, 4).unwrap();
        assert_eq!(test.len(), 3);
        assert_eq!(train.len(), 7);
        let mut all: Vec<usize> = train.iter().chain(&test).copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
        assert_eq!(train_test_split(10, 0.3, 4).unwrap(), (train, test));
        assert_ne!(
            train_test_split(10, 0.3, 5).unwrap().1,
            (train_test_split(10, 0.3, 4).unwrap().1)
        );
    }

    #[test]
    fn split_rejects_bad_sizes() {
        assert!(train_test_split(1, 0.5, 0).is_err());
        assert!(train_test_split(10, 0.0, 0).is_err());
        assert!(train_test_split(10, 1.0, 0).is_err());
    }

    #[test]
    fn kfold_tests_cover_every_row_exactly_once() {
        let folds = KFold::new(3).with_shuffle(11).split(10).unwrap();
        let mut seen = Vec::new();
        for (train, test) in &folds {
            assert_eq!(train.len() + test.len(), 10);
            seen.extend(test.iter().copied());
        }
        seen.sort_unstable();
        assert_eq!(seen, (0..10).collect::<Vec<_>>());
        // 10 rows over 3 folds: sizes 4, 3, 3.
        let sizes: Vec<usize> = folds.iter().map(|(_, t)| t.len()).collect();
        assert_eq!(sizes, vec![4, 3, 3]);
    }

    #[test]
    fn kfold_without_shuffle_is_in_row_order() {
        let folds = KFold::new(2).split(4).unwrap();
        assert_eq!(folds[0].1, vec![0, 1]);
        assert_eq!(folds[1].1, vec![2, 3]);
        assert!(KFold::new(5).split(4).is_err());
    }

    #[test]
    fn select_rows_copies_in_the_order_given() {
        let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = DenseView::new(&data, 3, 2).unwrap();
        let picked = select_rows(&view, &[2, 0]).unwrap();
        assert_eq!(picked.into_vec(), vec![5.0, 6.0, 1.0, 2.0]);
        assert!(select_rows(&view, &[3]).is_err());
    }

    /// A model that predicts the mean of its training targets.
    struct MeanModel(f64);

    impl SupervisedEstimator for MeanModel {
        type Float = f64;
        type FitError = Error;
        fn fit<X, Y>(&mut self, _x: X, y: Y) -> Result<(), Error>
        where
            X: ArrayView<Elem = f64>,
            Y: crate::TargetView<Elem = f64>,
        {
            let n = y.len();
            self.0 = (0..n).map(|i| y.get(i)).sum::<f64>() / n as f64;
            Ok(())
        }
    }

    #[test]
    fn cross_validation_returns_one_score_per_fold() {
        let x: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let y: Vec<f64> = vec![2.0; 8];
        let view = DenseView::new(&x, 8, 1).unwrap();
        let scores = cross_val_score(
            &view,
            &y,
            &KFold::new(4),
            || MeanModel(0.0),
            |m, _x, ty| {
                // Score: negative absolute error of the constant prediction.
                Ok(-ty.iter().map(|t| (t - m.0).abs()).sum::<f64>())
            },
        )
        .unwrap();
        assert_eq!(scores, vec![0.0; 4]);
    }
}
