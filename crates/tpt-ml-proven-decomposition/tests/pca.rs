// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! PCA behaviour: known answers, orthonormality, reconstruction, degenerate
//! inputs, parameter errors, and element-type agreement.

use tpt_ml_proven_core::{ArrayView, DenseView, Error, Transformer, UnsupervisedEstimator};
use tpt_ml_proven_decomposition::Pca;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn random_matrix(seed: u64, rows: usize, cols: usize) -> Vec<f64> {
    let mut g = Lcg(seed);
    (0..rows * cols).map(|_| g.next() * 4.0 - 2.0).collect()
}

fn fit(data: &[f64], rows: usize, cols: usize, k: usize) -> Pca<f64> {
    let mut p = Pca::<f64>::new(k);
    UnsupervisedEstimator::fit(&mut p, DenseView::new(data, rows, cols).unwrap()).unwrap();
    p
}

#[test]
fn points_on_a_line_give_that_line_as_first_component() {
    // y = 2x exactly, so the first principal axis is (1, 2) / sqrt(5).
    let data = [-2.0_f64, -4.0, -1.0, -2.0, 0.0, 0.0, 1.0, 2.0, 2.0, 4.0];
    let p = fit(&data, 5, 2, 1);
    let c = p.components().unwrap();
    let inv = 1.0 / 5.0_f64.sqrt();
    // Sign convention: the largest-magnitude entry (here y) is positive.
    assert!((c.get(0, 0) - inv).abs() < 1e-9, "{}", c.get(0, 0));
    assert!((c.get(0, 1) - 2.0 * inv).abs() < 1e-9, "{}", c.get(0, 1));
    assert!((p.explained_variance_ratio().unwrap()[0] - 1.0).abs() < 1e-9);
}

#[test]
fn components_are_orthonormal_on_random_data() {
    for seed in 0..10_u64 {
        let (rows, cols) = (30, 4);
        let data = random_matrix(seed, rows, cols);
        let p = fit(&data, rows, cols, cols);
        let c = p.components().unwrap();
        for i in 0..cols {
            let norm: f64 = (0..cols).map(|j| c.get(i, j) * c.get(i, j)).sum();
            assert!((norm - 1.0).abs() < 1e-9, "seed {seed}: norm {norm}");
            for l in i + 1..cols {
                let dot: f64 = (0..cols).map(|j| c.get(i, j) * c.get(l, j)).sum();
                assert!(dot.abs() < 1e-9, "seed {seed}: dot {dot}");
            }
        }
    }
}

#[test]
fn explained_variance_is_sorted_and_ratios_sum_to_at_most_one() {
    let data = random_matrix(4, 40, 5);
    let p = fit(&data, 40, 5, 5);
    let ev = p.explained_variance().unwrap();
    assert!(ev.windows(2).all(|w| w[0] >= w[1]));
    let total = p.total_variance().unwrap();
    let sum: f64 = ev.iter().sum();
    assert!(
        (sum - total).abs() < 1e-9 * (1.0 + total),
        "all components keep all variance"
    );
    let ratio_sum: f64 = p.explained_variance_ratio().unwrap().iter().sum();
    assert!((ratio_sum - 1.0).abs() < 1e-9);
}

#[test]
fn full_rank_projection_round_trips() {
    let (rows, cols) = (25, 3);
    let data = random_matrix(8, rows, cols);
    let mut p = Pca::<f64>::new(cols);
    let x = DenseView::new(&data, rows, cols).unwrap();
    let z = p.fit_transform(x).unwrap();
    let back = p.inverse_transform(&z.view()).unwrap();
    for (a, b) in back.into_vec().iter().zip(&data) {
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }
}

#[test]
fn scores_are_centred() {
    let (rows, cols) = (20, 2);
    let data = random_matrix(9, rows, cols);
    let mut p = Pca::<f64>::new(2);
    let z = p
        .fit_transform(DenseView::new(&data, rows, cols).unwrap())
        .unwrap();
    for c in 0..2 {
        let mean: f64 = (0..rows).map(|r| z.get(r, c)).sum::<f64>() / rows as f64;
        assert!(mean.abs() < 1e-9);
    }
}

#[test]
fn constant_columns_fit_without_nan_and_have_zero_ratio_when_nothing_varies() {
    let data = [3.0_f64, 7.0, 3.0, 7.0, 3.0, 7.0];
    let p = fit(&data, 3, 2, 2);
    assert_eq!(p.total_variance(), Some(0.0));
    assert!(p
        .explained_variance_ratio()
        .unwrap()
        .iter()
        .all(|&r| r == 0.0));
    assert!(p
        .components()
        .unwrap()
        .clone()
        .into_vec()
        .iter()
        .all(|v| v.is_finite()));
}

#[test]
fn rank_deficient_data_keeps_one_direction() {
    // Second column is twice the first, so the rank is one.
    let data = [1.0_f64, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0];
    let p = fit(&data, 4, 2, 2);
    let ev = p.explained_variance().unwrap();
    assert!(
        ev[1].abs() < 1e-9,
        "second component carries no variance: {}",
        ev[1]
    );
    let ratio = p.explained_variance_ratio().unwrap();
    assert!((ratio[0] - 1.0).abs() < 1e-9);
}

#[test]
fn n_components_bounds_are_reported_as_parameter_errors() {
    let data = random_matrix(1, 5, 3);
    let view = || DenseView::new(&data, 5, 3).unwrap();
    for k in [0_usize, 4] {
        let mut p = Pca::<f64>::new(k);
        let err = UnsupervisedEstimator::fit(&mut p, view()).unwrap_err();
        assert!(
            matches!(
                err,
                Error::InvalidParameter {
                    name: "n_components",
                    ..
                }
            ),
            "k = {k}: {err:?}"
        );
    }
}

#[test]
fn too_few_samples_and_nan_are_rejected() {
    let one = [1.0_f64, 2.0];
    let mut p = Pca::<f64>::new(1);
    let err = UnsupervisedEstimator::fit(&mut p, DenseView::new(&one, 1, 2).unwrap()).unwrap_err();
    assert!(matches!(err, Error::InvalidParameter { name: "rows", .. }));

    let bad = [1.0_f64, f64::NAN, 2.0, 3.0];
    let mut p = Pca::<f64>::new(1);
    let err = UnsupervisedEstimator::fit(&mut p, DenseView::new(&bad, 2, 2).unwrap()).unwrap_err();
    assert_eq!(err, Error::NonFinite { what: "X" });
    assert!(p.components().is_none());
}

#[test]
fn transform_before_fit_and_feature_mismatch_are_errors() {
    let p = Pca::<f64>::new(1);
    let data = [0.0_f64, 1.0];
    assert_eq!(
        p.transform(DenseView::new(&data, 2, 1).unwrap())
            .unwrap_err(),
        Error::NotFitted
    );

    let train = random_matrix(2, 6, 3);
    let fitted = fit(&train, 6, 3, 2);
    let wrong = [1.0_f64, 2.0];
    let err = fitted
        .transform(DenseView::new(&wrong, 1, 2).unwrap())
        .unwrap_err();
    assert!(matches!(err, Error::Shape(_)));
}

#[test]
fn f32_matches_f64_to_single_precision() {
    let (rows, cols) = (30, 3);
    let data64 = random_matrix(5, rows, cols);
    let data32: Vec<f32> = data64.iter().map(|&v| v as f32).collect();
    let p64 = fit(&data64, rows, cols, 2);
    let mut p32 = Pca::<f32>::new(2);
    UnsupervisedEstimator::fit(&mut p32, DenseView::new(&data32, rows, cols).unwrap()).unwrap();
    let (c64, c32) = (p64.components().unwrap(), p32.components().unwrap());
    for i in 0..2 {
        for j in 0..cols {
            assert!((c64.get(i, j) - f64::from(c32.get(i, j))).abs() < 1e-4);
        }
    }
}

#[test]
fn property_random_shapes_stay_valid() {
    for seed in 0..30_u64 {
        let rows = 2 + (seed as usize % 12);
        let cols = 1 + (seed as usize % 4);
        let k = 1 + (seed as usize % cols.min(rows));
        // Integer data gives repeated values and degenerate directions.
        let data: Vec<f64> = random_matrix(seed + 50, rows, cols)
            .iter()
            .map(|v| v.round())
            .collect();
        let p = fit(&data, rows, cols, k);
        assert!(p
            .components()
            .unwrap()
            .clone()
            .into_vec()
            .iter()
            .all(|v| v.is_finite()));
        let z = p
            .transform(DenseView::new(&data, rows, cols).unwrap())
            .unwrap();
        assert_eq!((z.rows(), z.cols()), (rows, k));
        assert!(z.clone().into_vec().iter().all(|v| v.is_finite()));
    }
}
