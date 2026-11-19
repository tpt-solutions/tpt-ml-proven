// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! `score` on the supervised linear models: R² for regressors, accuracy for classifiers.

use tpt_ml_proven_core::{DenseView, SupervisedEstimator};
use tpt_ml_proven_linear::{LogisticRegression, Ridge};

#[test]
fn ridge_scores_near_one_on_exactly_linear_data() {
    // y = 2·x0 − x1 + 3, exactly, so a nearly unpenalised ridge fits it perfectly.
    let x = [
        0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 1.0,
    ];
    let y: Vec<f64> = x.chunks(2).map(|r| 2.0 * r[0] - r[1] + 3.0).collect();
    let view = DenseView::new(&x, 6, 2).unwrap();
    let mut m = Ridge::<f64>::new(1e-9);
    SupervisedEstimator::fit(&mut m, view, &y[..]).unwrap();
    let r2 = m.score(DenseView::new(&x, 6, 2).unwrap(), &y).unwrap();
    assert!((r2 - 1.0).abs() < 1e-9, "r2 = {r2}");
}

#[test]
fn logistic_scores_one_on_separable_labels() {
    let x = [-2.0_f64, -1.5, -1.0, -0.5, 1.0, 0.5, 2.0, 1.5];
    let y = [0.0_f64, 0.0, 1.0, 1.0];
    let view = DenseView::new(&x, 4, 2).unwrap();
    let mut m = LogisticRegression::<f64>::new(10.0);
    SupervisedEstimator::fit(&mut m, view, &y[..]).unwrap();
    let acc = m.score(DenseView::new(&x, 4, 2).unwrap(), &y).unwrap();
    assert_eq!(acc, 1.0);
}
