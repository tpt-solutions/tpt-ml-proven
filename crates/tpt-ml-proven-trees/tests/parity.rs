// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Parity with scikit-learn's `GradientBoostingRegressor` on the `gbm_regression` fixture
//! (squared error, friedman_mse splits, no subsampling, no feature subsampling).

use std::collections::HashMap;
use std::path::PathBuf;

use tpt_ml_proven_core::{DenseView, Predictor, SupervisedEstimator};
use tpt_ml_proven_trees::GradientBoostingRegressor;

fn load(name: &str) -> HashMap<String, Vec<String>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/parity/fixtures")
        .join(format!("{name}.txt"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|line| {
            let mut parts = line.split_whitespace();
            let key = parts.next().unwrap().to_string();
            (key, parts.map(str::to_string).collect())
        })
        .collect()
}

fn floats(fx: &HashMap<String, Vec<String>>, key: &str) -> Vec<f64> {
    fx[key].iter().map(|s| s.parse().unwrap()).collect()
}

#[test]
fn boosted_predictions_match_scikit_learn() {
    let fx = load("gbm_regression");
    let rows: usize = fx["rows"][0].parse().unwrap();
    let cols: usize = fx["cols"][0].parse().unwrap();
    let n_estimators: usize = fx["n_estimators"][0].parse().unwrap();
    let learning_rate: f64 = fx["learning_rate"][0].parse().unwrap();
    let max_depth: usize = fx["max_depth"][0].parse().unwrap();
    let min_leaf: usize = fx["min_samples_leaf"][0].parse().unwrap();
    let x = floats(&fx, "x");
    let y = floats(&fx, "y");

    let mut m = GradientBoostingRegressor::new(n_estimators, max_depth)
        .with_learning_rate(learning_rate)
        .with_min_samples_leaf(min_leaf);
    SupervisedEstimator::fit(&mut m, DenseView::new(&x, rows, cols).unwrap(), &y[..]).unwrap();

    let pred = Predictor::predict(&m, DenseView::new(&x, rows, cols).unwrap()).unwrap();
    for (i, (&p, &q)) in pred.iter().zip(&floats(&fx, "pred")).enumerate() {
        assert!(
            (p - q).abs() <= 1e-9 * (1.0 + q.abs()),
            "gbm prediction {i}: {p} vs scikit-learn {q}"
        );
    }
}
