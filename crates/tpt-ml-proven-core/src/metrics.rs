// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Scoring functions for regression, classification and clustering.
//!
//! The definitions follow scikit-learn's `sklearn.metrics`. Results are `f64` whatever the
//! input type. `no_std` has no `sqrt` or `ln`, so both are implemented here, to full `f64`
//! accuracy for the finite positive inputs they receive.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::error::{Error, ShapeError};
use crate::{ArrayView, Float};

fn shape(rows: usize, cols: usize, reason: &'static str) -> Error {
    Error::Shape(ShapeError { rows, cols, reason })
}

fn check_pair(a: usize, b: usize) -> Result<(), Error> {
    if a == 0 {
        return Err(Error::EmptyInput);
    }
    if a != b {
        return Err(shape(a, 1, "true and predicted lengths differ"));
    }
    Ok(())
}

/// Square root by Newton's method, for finite `x >= 0`.
pub(crate) fn sqrt(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut g = if x < 1.0 { 1.0 } else { x };
    for _ in 0..200 {
        let next = 0.5 * (g + x / g);
        if next == g {
            break;
        }
        g = next;
    }
    g
}

/// Natural logarithm for finite `x > 0`: scale `x` into `[1, 2)`, then use the atanh series.
pub(crate) fn ln(x: f64) -> f64 {
    const LN_2: f64 = core::f64::consts::LN_2;
    let mut m = x;
    let mut k = 0_i32;
    while m >= 2.0 {
        m *= 0.5;
        k += 1;
    }
    while m < 1.0 {
        m *= 2.0;
        k -= 1;
    }
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut term = s;
    let mut sum = 0.0;
    for i in 0..80 {
        sum += term / (2 * i + 1) as f64;
        term *= s2;
    }
    2.0 * sum + f64::from(k) * LN_2
}

/// Mean squared error.
pub fn mse<F: Float>(y_true: &[F], y_pred: &[F]) -> Result<f64, Error> {
    check_pair(y_true.len(), y_pred.len())?;
    let sum: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(t, p)| {
            let d = t.to_f64() - p.to_f64();
            d * d
        })
        .sum();
    Ok(sum / y_true.len() as f64)
}

/// Root mean squared error.
pub fn rmse<F: Float>(y_true: &[F], y_pred: &[F]) -> Result<f64, Error> {
    mse(y_true, y_pred).map(sqrt)
}

/// Mean absolute error.
pub fn mae<F: Float>(y_true: &[F], y_pred: &[F]) -> Result<f64, Error> {
    check_pair(y_true.len(), y_pred.len())?;
    let sum: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(t, p)| (t.to_f64() - p.to_f64()).abs())
        .sum();
    Ok(sum / y_true.len() as f64)
}

/// Coefficient of determination, `1 − SS_res / SS_tot`.
///
/// When the true values are constant, `SS_tot` is zero: the score is `1.0` for a perfect
/// prediction and `0.0` otherwise, as in scikit-learn.
pub fn r2<F: Float>(y_true: &[F], y_pred: &[F]) -> Result<f64, Error> {
    check_pair(y_true.len(), y_pred.len())?;
    let n = y_true.len() as f64;
    let mean = y_true.iter().map(|t| t.to_f64()).sum::<f64>() / n;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for (t, p) in y_true.iter().zip(y_pred) {
        let (t, p) = (t.to_f64(), p.to_f64());
        ss_res += (t - p) * (t - p);
        ss_tot += (t - mean) * (t - mean);
    }
    if ss_tot == 0.0 {
        return Ok(if ss_res == 0.0 { 1.0 } else { 0.0 });
    }
    Ok(1.0 - ss_res / ss_tot)
}

/// Fraction of labels predicted correctly.
pub fn accuracy<L: PartialEq>(y_true: &[L], y_pred: &[L]) -> Result<f64, Error> {
    check_pair(y_true.len(), y_pred.len())?;
    let hits = y_true.iter().zip(y_pred).filter(|(t, p)| t == p).count();
    Ok(hits as f64 / y_true.len() as f64)
}

/// Probabilities are clipped to `[EPS, 1 − EPS]` before taking logarithms, as in scikit-learn.
const EPS: f64 = 1e-15;

/// Binary log loss. `y_true` holds 0 or 1, and `p_positive` is the predicted probability of
/// class 1 for each row.
pub fn log_loss_binary<F: Float>(y_true: &[F], p_positive: &[F]) -> Result<f64, Error> {
    check_pair(y_true.len(), p_positive.len())?;
    let mut total = 0.0;
    for (t, p) in y_true.iter().zip(p_positive) {
        let t = t.to_f64();
        if t != 0.0 && t != 1.0 {
            return Err(Error::InvalidParameter {
                name: "y_true",
                reason: "binary log loss needs labels 0 or 1",
            });
        }
        let p = p.to_f64().clamp(EPS, 1.0 - EPS);
        total -= t * ln(p) + (1.0 - t) * ln(1.0 - p);
    }
    Ok(total / y_true.len() as f64)
}

/// Multiclass log loss. `labels[i]` is the true class index of row `i`, and `proba` holds one
/// row of class probabilities per sample (`rows × classes`). Each row is renormalised to sum
/// to one before the loss is taken.
pub fn log_loss<F, Y>(labels: &[usize], proba: &Y) -> Result<f64, Error>
where
    F: Float,
    Y: ArrayView<Elem = F>,
{
    if labels.is_empty() {
        return Err(Error::EmptyInput);
    }
    if proba.rows() != labels.len() {
        return Err(shape(
            proba.rows(),
            proba.cols(),
            "probability rows differ from the number of labels",
        ));
    }
    let mut total = 0.0;
    for (r, &label) in labels.iter().enumerate() {
        if label >= proba.cols() {
            return Err(shape(
                proba.rows(),
                proba.cols(),
                "a label is outside the probability columns",
            ));
        }
        let row_sum: f64 = (0..proba.cols()).map(|c| proba.get(r, c).to_f64()).sum();
        if row_sum <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "proba",
                reason: "a probability row sums to zero",
            });
        }
        let p = (proba.get(r, label).to_f64() / row_sum).clamp(EPS, 1.0 - EPS);
        total -= ln(p);
    }
    Ok(total / labels.len() as f64)
}

/// Mean silhouette coefficient over all samples, with Euclidean distance.
///
/// Needs between 2 and `n − 1` distinct labels. A sample alone in its cluster scores 0.
pub fn silhouette<F, X>(x: &X, labels: &[usize]) -> Result<f64, Error>
where
    F: Float,
    X: ArrayView<Elem = F>,
{
    let n = x.rows();
    if n == 0 {
        return Err(Error::EmptyInput);
    }
    if labels.len() != n {
        return Err(shape(
            n,
            x.cols(),
            "label count differs from the number of rows",
        ));
    }
    let mut clusters: Vec<usize> = labels.to_vec();
    clusters.sort_unstable();
    clusters.dedup();
    let k = clusters.len();
    if k < 2 || k > n - 1 {
        return Err(Error::InvalidParameter {
            name: "labels",
            reason: "silhouette needs between 2 and n-1 distinct clusters",
        });
    }
    let index_of = |label: usize| clusters.binary_search(&label).expect("label is a cluster");
    let mut sizes = alloc::vec![0_usize; k];
    for &l in labels {
        sizes[index_of(l)] += 1;
    }

    let mut total = 0.0;
    let mut dist_sum = alloc::vec![0.0_f64; k];
    for i in 0..n {
        dist_sum.iter_mut().for_each(|d| *d = 0.0);
        for j in 0..n {
            if i == j {
                continue;
            }
            let d2: f64 = (0..x.cols())
                .map(|c| {
                    let d = x.get(i, c).to_f64() - x.get(j, c).to_f64();
                    d * d
                })
                .sum();
            dist_sum[index_of(labels[j])] += sqrt(d2);
        }
        let own = index_of(labels[i]);
        if sizes[own] == 1 {
            continue; // silhouette of a singleton is 0
        }
        let a = dist_sum[own] / (sizes[own] - 1) as f64;
        let b = (0..k)
            .filter(|&c| c != own)
            .map(|c| dist_sum[c] / sizes[c] as f64)
            .fold(f64::INFINITY, f64::min);
        let denom = a.max(b);
        if denom > 0.0 {
            total += (b - a) / denom;
        }
    }
    Ok(total / n as f64)
}

/// Adjusted Rand index between two labelings of the same samples. 1 is identical up to
/// renaming, and values near 0 are what chance gives.
pub fn adjusted_rand(labels_true: &[usize], labels_pred: &[usize]) -> Result<f64, Error> {
    check_pair(labels_true.len(), labels_pred.len())?;
    let n = labels_true.len();
    let mut cells: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let mut rows: BTreeMap<usize, usize> = BTreeMap::new();
    let mut cols: BTreeMap<usize, usize> = BTreeMap::new();
    for (&a, &b) in labels_true.iter().zip(labels_pred) {
        *cells.entry((a, b)).or_insert(0) += 1;
        *rows.entry(a).or_insert(0) += 1;
        *cols.entry(b).or_insert(0) += 1;
    }
    let choose2 = |m: usize| (m as f64) * (m as f64 - 1.0) / 2.0;
    let index: f64 = cells.values().map(|&m| choose2(m)).sum();
    let sum_rows: f64 = rows.values().map(|&m| choose2(m)).sum();
    let sum_cols: f64 = cols.values().map(|&m| choose2(m)).sum();
    let total = choose2(n);
    if total == 0.0 {
        return Ok(1.0);
    }
    let expected = sum_rows * sum_cols / total;
    let max_index = 0.5 * (sum_rows + sum_cols);
    if (max_index - expected).abs() < f64::EPSILON {
        // Both labelings are trivial (one cluster, or all singletons): identical by definition.
        return Ok(1.0);
    }
    Ok((index - expected) / (max_index - expected))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DenseView;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn sqrt_and_ln_match_known_values() {
        assert!(close(sqrt(2.0), core::f64::consts::SQRT_2));
        assert!(close(sqrt(1e-8), 1e-4));
        assert!(close(sqrt(1e12), 1e6));
        assert!(close(ln(1.0), 0.0));
        assert!(close(ln(core::f64::consts::E), 1.0));
        assert!(close(ln(1e-15), -34.538776394910684));
        assert!(close(ln(0.5), -core::f64::consts::LN_2));
    }

    #[test]
    fn regression_errors_on_a_known_example() {
        let t = [3.0_f64, -0.5, 2.0, 7.0];
        let p = [2.5_f64, 0.0, 2.0, 8.0];
        assert!(close(mse(&t, &p).unwrap(), 0.375));
        assert!(close(rmse(&t, &p).unwrap(), 0.375_f64.sqrt()));
        assert!(close(mae(&t, &p).unwrap(), 0.5));
        // ss_res = 0.25 + 0.25 + 0 + 1 = 1.5. The mean is 2.875, so ss_tot is
        // 0.015625 + 11.390625 + 0.765625 + 17.015625 = 29.1875.
        let r2v = r2(&t, &p).unwrap();
        assert!(close(r2v, 1.0 - 1.5 / 29.1875));
    }

    #[test]
    fn r2_is_one_for_perfect_and_handles_constant_truth() {
        let t = [1.0_f64, 2.0, 3.0];
        assert!(close(r2(&t, &t).unwrap(), 1.0));
        let c = [5.0_f64, 5.0, 5.0];
        assert_eq!(r2(&c, &c).unwrap(), 1.0);
        assert_eq!(r2(&c, &[4.0, 5.0, 5.0]).unwrap(), 0.0);
    }

    #[test]
    fn mismatched_or_empty_inputs_are_errors() {
        let a = [1.0_f64, 2.0];
        assert!(matches!(mse(&a, &a[..1]), Err(Error::Shape(_))));
        assert_eq!(mae::<f64>(&[], &[]).unwrap_err(), Error::EmptyInput);
    }

    #[test]
    fn accuracy_counts_matches() {
        assert!(close(accuracy(&[1, 0, 1, 1], &[1, 1, 1, 0]).unwrap(), 0.5));
    }

    #[test]
    fn binary_log_loss_matches_the_formula() {
        let y = [1.0_f64, 0.0];
        let p = [0.8_f64, 0.3];
        let want = -(0.8_f64.ln() + 0.7_f64.ln()) / 2.0;
        assert!(close(log_loss_binary(&y, &p).unwrap(), want));
        assert!(log_loss_binary(&[2.0_f64], &[0.5]).is_err());
    }

    #[test]
    fn multiclass_log_loss_agrees_with_binary() {
        let labels = [0_usize, 1];
        let proba = [0.8_f64, 0.2, 0.3, 0.7];
        let view = DenseView::new(&proba, 2, 2).unwrap();
        let want = -(0.8_f64.ln() + 0.7_f64.ln()) / 2.0;
        assert!(close(log_loss(&labels, &view).unwrap(), want));
    }

    #[test]
    fn silhouette_is_one_for_well_separated_clusters() {
        let x = [0.0_f64, 0.0, 0.1, 0.0, 10.0, 10.0, 10.1, 10.0];
        let view = DenseView::new(&x, 4, 2).unwrap();
        let s = silhouette(&view, &[0, 0, 1, 1]).unwrap();
        assert!(s > 0.98, "got {s}");
    }

    #[test]
    fn silhouette_rejects_one_cluster_or_all_singletons() {
        let x = [0.0_f64, 1.0, 2.0];
        let view = DenseView::new(&x, 3, 1).unwrap();
        assert!(silhouette(&view, &[0, 0, 0]).is_err());
        assert!(silhouette(&view, &[0, 1, 2]).is_err());
    }

    #[test]
    fn adjusted_rand_is_one_for_renamed_partitions_and_zero_ish_for_mixing() {
        assert!(close(
            adjusted_rand(&[0, 0, 1, 1], &[5, 5, 9, 9]).unwrap(),
            1.0
        ));
        // Index 2, expected 2.8, maximum 6.5: (2 − 2.8) / (6.5 − 2.8). Below chance, so negative.
        let ari = adjusted_rand(&[0, 1, 0, 1, 0, 1], &[0, 0, 1, 1, 0, 0]).unwrap();
        assert!(close(ari, -0.8 / 3.7), "got {ari}");
    }
}
