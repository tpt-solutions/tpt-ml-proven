// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Parity with scikit-learn's Ridge, Lasso and ElasticNet, on fixtures from
//! `tools/parity/generate_fixtures.py`. Both sides run to a tight tolerance, so
//! the coefficients should agree closely. The comparison is looser than the 1e-9 used
//! for KMeans and PCA, because the coordinate-descent stopping rules differ slightly.

use std::collections::HashMap;
use std::path::PathBuf;

use tpt_ml_proven_core::{DenseView, Predictor, SupervisedEstimator};
use tpt_ml_proven_linear::{ElasticNet, Lasso, Ridge};

const TOL: f64 = 1e-6;

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

fn assert_close(actual: f64, expected: f64, what: &str) {
    let scale = 1.0 + expected.abs();
    assert!(
        (actual - expected).abs() <= TOL * scale,
        "{what}: {actual} vs scikit-learn {expected}"
    );
}

fn check(name: &str, fit: impl FnOnce(&[f64], &[f64], usize, usize) -> (Vec<f64>, f64)) {
    let fx = load(name);
    let rows: usize = fx["rows"][0].parse().unwrap();
    let cols: usize = fx["cols"][0].parse().unwrap();
    let x = floats(&fx, "x");
    let y = floats(&fx, "y");
    let (coef, intercept) = fit(&x, &y, rows, cols);
    for (j, &w) in floats(&fx, "coef").iter().enumerate() {
        assert_close(coef[j], w, &format!("{name}: coef[{j}]"));
    }
    assert_close(
        intercept,
        floats(&fx, "intercept")[0],
        &format!("{name}: intercept"),
    );
}

#[test]
fn ridge_matches_scikit_learn() {
    check("linear_ridge", |x, y, rows, cols| {
        let mut m = Ridge::<f64>::new(1.0);
        SupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols).unwrap(), y).unwrap();
        (m.coef().unwrap().to_vec(), m.intercept().unwrap())
    });
}

#[test]
fn lasso_matches_scikit_learn() {
    check("linear_lasso", |x, y, rows, cols| {
        let mut m = Lasso::<f64>::new(0.05)
            .with_tol(1e-12)
            .with_max_iter(100_000);
        SupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols).unwrap(), y).unwrap();
        (m.coef().unwrap().to_vec(), m.intercept().unwrap())
    });
}

#[test]
fn elasticnet_matches_scikit_learn() {
    check("linear_elasticnet", |x, y, rows, cols| {
        let mut m = ElasticNet::<f64>::new(0.05, 0.5)
            .with_tol(1e-12)
            .with_max_iter(100_000);
        SupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols).unwrap(), y).unwrap();
        (m.coef().unwrap().to_vec(), m.intercept().unwrap())
    });
}

#[test]
fn predictions_follow_the_fitted_coefficients() {
    let fx = load("linear_ridge");
    let rows: usize = fx["rows"][0].parse().unwrap();
    let cols: usize = fx["cols"][0].parse().unwrap();
    let x = floats(&fx, "x");
    let y = floats(&fx, "y");
    let mut m = Ridge::<f64>::new(1.0);
    SupervisedEstimator::fit(&mut m, DenseView::new(&x, rows, cols).unwrap(), &y[..]).unwrap();
    let pred = Predictor::predict(&m, DenseView::new(&x, rows, cols).unwrap()).unwrap();
    let coef = m.coef().unwrap();
    let b = m.intercept().unwrap();
    for r in 0..rows {
        let manual: f64 = (0..cols).map(|c| x[r * cols + c] * coef[c]).sum::<f64>() + b;
        assert!((pred[r] - manual).abs() < 1e-12);
    }
}
