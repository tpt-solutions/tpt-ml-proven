// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! DBSCAN (Ester et al., 1996).
//!
//! A sample is a *core* sample when at least `min_samples` samples (itself included)
//! lie within distance `eps`. Clusters grow from core samples through their
//! neighbourhoods. A sample that is not core but is reached from one is a *border*
//! sample, and it takes the first cluster that reaches it. Samples reached by none are
//! *noise*, labelled `-1`.
//!
//! The labelling order matches scikit-learn's implementation, so labels can be compared
//! exactly. Neighbour search is brute force, `O(n²·d)`, with no index structure.

use tpt_ml_proven_core::error::Error;
use tpt_ml_proven_core::{ArrayView, Float, UnsupervisedEstimator};

use crate::kmeans::{for_each_row, sq_dist};

/// Label given to noise samples.
pub const NOISE: i64 = -1;

/// DBSCAN estimator. Build with [`Dbscan::new`], then call `fit`.
#[derive(Debug, Clone, PartialEq)]
pub struct Dbscan<F: Float> {
    eps: F,
    min_samples: usize,
    labels: Option<Vec<i64>>,
    core: Option<Vec<usize>>,
}

impl<F: Float> Dbscan<F> {
    /// Unfitted DBSCAN with neighbourhood radius `eps` and `min_samples`.
    pub fn new(eps: F, min_samples: usize) -> Self {
        Self {
            eps,
            min_samples,
            labels: None,
            core: None,
        }
    }

    /// Cluster label of each sample, `-1` for noise, or `None` before `fit`.
    pub fn labels(&self) -> Option<&[i64]> {
        self.labels.as_deref()
    }

    /// Indices of the core samples, in increasing order, or `None` before `fit`.
    pub fn core_sample_indices(&self) -> Option<&[usize]> {
        self.core.as_deref()
    }

    /// Number of clusters found (excluding noise).
    pub fn n_clusters(&self) -> Option<usize> {
        self.labels.as_ref().map(|l| {
            l.iter()
                .copied()
                .max()
                .map_or(0, |m| (m + 1).max(0) as usize)
        })
    }
}

/// Every sample within `eps` of sample `i`, including `i` itself.
fn neighbours<F: Float>(data: &[F], rows: usize, cols: usize, i: usize, eps2: F) -> Vec<usize> {
    let a = &data[i * cols..(i + 1) * cols];
    (0..rows)
        .filter(|&j| sq_dist(a, &data[j * cols..(j + 1) * cols]) <= eps2)
        .collect()
}

impl<F: Float> UnsupervisedEstimator for Dbscan<F> {
    type Float = F;
    type FitError = Error;

    fn fit<X>(&mut self, x: X) -> Result<(), Error>
    where
        X: ArrayView<Elem = F>,
    {
        self.labels = None;
        self.core = None;
        let (rows, cols) = (x.rows(), x.cols());
        if rows == 0 || cols == 0 {
            return Err(Error::EmptyInput);
        }
        if !self.eps.is_finite() || self.eps <= F::ZERO {
            return Err(Error::InvalidParameter {
                name: "eps",
                reason: "must be finite and positive",
            });
        }
        if self.min_samples == 0 {
            return Err(Error::InvalidParameter {
                name: "min_samples",
                reason: "must be at least 1",
            });
        }
        if !x.all_finite() {
            return Err(Error::NonFinite { what: "X" });
        }

        let mut data = Vec::with_capacity(rows * cols);
        for_each_row(&x, |_, row| data.extend_from_slice(row));
        let eps2 = self.eps * self.eps;

        let is_core: Vec<bool> = (0..rows)
            .map(|i| neighbours(&data, rows, cols, i, eps2).len() >= self.min_samples)
            .collect();

        let mut labels = vec![NOISE; rows];
        let mut next_label: i64 = 0;
        for i in 0..rows {
            if labels[i] != NOISE || !is_core[i] {
                continue;
            }
            // Grow a cluster from core sample i. Labels are assigned when a sample is
            // first reached, and only core samples are expanded.
            labels[i] = next_label;
            let mut stack = vec![i];
            while let Some(j) = stack.pop() {
                for k in neighbours(&data, rows, cols, j, eps2) {
                    if labels[k] == NOISE {
                        labels[k] = next_label;
                        if is_core[k] {
                            stack.push(k);
                        }
                    }
                }
            }
            next_label += 1;
        }

        let core = (0..rows).filter(|&i| is_core[i]).collect();
        self.labels = Some(labels);
        self.core = Some(core);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ml_proven_core::DenseView;

    fn fit(data: &[f64], rows: usize, cols: usize, eps: f64, min: usize) -> Dbscan<f64> {
        let mut db = Dbscan::new(eps, min);
        UnsupervisedEstimator::fit(&mut db, DenseView::new(data, rows, cols).unwrap()).unwrap();
        db
    }

    #[test]
    fn two_dense_groups_and_an_outlier() {
        let data = [
            0.0_f64, 0.0, 0.1, 0.0, 0.0, 0.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1, 20.0, 20.0,
        ];
        let db = fit(&data, 7, 2, 0.5, 3);
        assert_eq!(db.labels().unwrap(), &[0, 0, 0, 1, 1, 1, NOISE]);
        assert_eq!(db.n_clusters(), Some(2));
        assert_eq!(db.core_sample_indices().unwrap(), &[0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn border_point_is_labelled_but_not_expanded() {
        // Distances are exact binary fractions, so the eps comparison is exact.
        // 0.0, 0.5 and 1.0 are core (at least three neighbours within eps = 1.0). 2.0 is
        // reached from core 1.0 but has only two neighbours, so it is a border sample.
        // 10.0 is noise.
        let data = [0.0_f64, 0.5, 1.0, 2.0, 10.0];
        let db = fit(&data, 5, 1, 1.0, 3);
        assert_eq!(db.labels().unwrap(), &[0, 0, 0, 0, NOISE]);
        assert_eq!(db.core_sample_indices().unwrap(), &[0, 1, 2]);
    }

    #[test]
    fn everything_is_noise_when_min_samples_is_unreachable() {
        let data = [0.0_f64, 1.0, 2.0];
        let db = fit(&data, 3, 1, 0.1, 2);
        assert_eq!(db.labels().unwrap(), &[NOISE, NOISE, NOISE]);
        assert_eq!(db.n_clusters(), Some(0));
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        let data = [0.0_f64, 1.0];
        let view = || DenseView::new(&data, 2, 1).unwrap();
        let mut db = Dbscan::new(0.0_f64, 1);
        assert!(matches!(
            UnsupervisedEstimator::fit(&mut db, view()),
            Err(Error::InvalidParameter { name: "eps", .. })
        ));
        let mut db = Dbscan::new(1.0_f64, 0);
        assert!(matches!(
            UnsupervisedEstimator::fit(&mut db, view()),
            Err(Error::InvalidParameter {
                name: "min_samples",
                ..
            })
        ));
    }
}
