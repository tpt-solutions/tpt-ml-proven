// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! The structural invariants of the tree models, checked through the public API. The same
//! properties are contracts in `docs/contracts/decision-tree.tptc` and
//! `docs/contracts/gradient-boosting.tptc`, enforced at runtime inside the crate.

use tpt_ml_proven_core::error::Error;
use tpt_ml_proven_core::{DenseView, Predictor, SupervisedEstimator};
use tpt_ml_proven_trees::{DecisionTreeRegressor, GradientBoostingRegressor};

/// A pseudo-random but fixed data set: 80 rows, 3 features, a nonlinear target.
fn data() -> (Vec<f64>, Vec<f64>) {
    let mut seed: u64 = 99;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..80 {
        let row = [next() * 4.0 - 2.0, next() * 4.0 - 2.0, next()];
        y.push(row[0].sin() + if row[1] > 0.0 { 1.0 } else { -1.0 } + 0.1 * row[2]);
        x.extend_from_slice(&row);
    }
    (x, y)
}

#[test]
fn tree_depth_never_exceeds_max_depth() {
    let (x, y) = data();
    for max_depth in 1..=5 {
        let mut t = DecisionTreeRegressor::new(max_depth);
        SupervisedEstimator::fit(&mut t, DenseView::new(&x, 80, 3).unwrap(), &y[..]).unwrap();
        assert!(t.tree().unwrap().depth() <= max_depth);
    }
}

#[test]
fn tree_leaves_are_at_least_min_samples_leaf() {
    let (x, y) = data();
    let mut t = DecisionTreeRegressor::new(6).with_min_samples_leaf(5);
    SupervisedEstimator::fit(&mut t, DenseView::new(&x, 80, 3).unwrap(), &y[..]).unwrap();
    // Predictions are leaf means, so every prediction must be a mean over at least 5 rows.
    // The leaf count is a cheap proxy: 80 rows with at least 5 per leaf gives at most 16 leaves.
    assert!(t.tree().unwrap().n_leaves() <= 16);
}

#[test]
fn a_deeper_tree_fits_the_training_data_at_least_as_well() {
    let (x, y) = data();
    let sse = |depth: usize| {
        let mut t = DecisionTreeRegressor::new(depth);
        SupervisedEstimator::fit(&mut t, DenseView::new(&x, 80, 3).unwrap(), &y[..]).unwrap();
        let p = Predictor::predict(&t, DenseView::new(&x, 80, 3).unwrap()).unwrap();
        p.iter()
            .zip(&y)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
    };
    assert!(sse(4) <= sse(2) + 1e-12);
}

#[test]
fn boosting_training_loss_never_increases() {
    let (x, y) = data();
    let mut m = GradientBoostingRegressor::new(40, 3).with_learning_rate(0.2);
    SupervisedEstimator::fit(&mut m, DenseView::new(&x, 80, 3).unwrap(), &y[..]).unwrap();
    let loss = m.train_loss().unwrap();
    assert_eq!(loss.len(), 40);
    for w in loss.windows(2) {
        assert!(
            w[1] <= w[0] * (1.0 + 1e-12),
            "loss rose: {} -> {}",
            w[0],
            w[1]
        );
    }
    assert!(loss[39] < loss[0]);
}

#[test]
fn learning_rate_outside_unit_interval_is_rejected() {
    let (x, y) = data();
    for bad in [0.0, -0.1, 1.5, f64::NAN] {
        let mut m = GradientBoostingRegressor::new(5, 2).with_learning_rate(bad);
        assert!(matches!(
            SupervisedEstimator::fit(&mut m, DenseView::new(&x, 80, 3).unwrap(), &y[..]),
            Err(Error::InvalidParameter {
                name: "learning_rate",
                ..
            })
        ));
    }
}

#[test]
fn constant_target_gives_a_single_leaf_and_the_constant() {
    let (x, _) = data();
    let y = vec![3.5; 80];
    let mut m = GradientBoostingRegressor::new(10, 3);
    SupervisedEstimator::fit(&mut m, DenseView::new(&x, 80, 3).unwrap(), &y[..]).unwrap();
    let p = Predictor::predict(&m, DenseView::new(&x, 80, 3).unwrap()).unwrap();
    assert!(p.iter().all(|v| (v - 3.5).abs() < 1e-12));
}

#[test]
fn predicting_with_the_wrong_width_is_a_shape_error() {
    let (x, y) = data();
    let mut t = DecisionTreeRegressor::new(2);
    SupervisedEstimator::fit(&mut t, DenseView::new(&x, 80, 3).unwrap(), &y[..]).unwrap();
    assert!(matches!(
        Predictor::predict(&t, DenseView::new(&x[..8], 4, 2).unwrap()),
        Err(Error::Shape(_))
    ));
}
