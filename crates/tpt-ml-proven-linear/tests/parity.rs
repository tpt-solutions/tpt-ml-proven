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

#[test]
fn logistic_matches_scikit_learn() {
    use tpt_ml_proven_linear::LogisticRegression;

    let fx = load("logistic_binary_c1");
    let rows: usize = fx["rows"][0].parse().unwrap();
    let cols: usize = fx["cols"][0].parse().unwrap();
    let x = floats(&fx, "x");
    let y = floats(&fx, "y");
    let c: f64 = fx["C"][0].parse().unwrap();

    let mut m = LogisticRegression::<f64>::new(c)
        .with_tol(1e-10)
        .with_max_iter(1000);
    SupervisedEstimator::fit(&mut m, DenseView::new(&x, rows, cols).unwrap(), &y[..]).unwrap();

    for (j, &w) in floats(&fx, "coef").iter().enumerate() {
        assert_close(m.coef().unwrap()[j], w, &format!("logistic: coef[{j}]"));
    }
    assert_close(
        m.intercept().unwrap(),
        floats(&fx, "intercept")[0],
        "logistic: intercept",
    );
    let proba = m
        .predict_proba(DenseView::new(&x, rows, cols).unwrap())
        .unwrap();
    for (i, (&p, &q)) in proba.iter().zip(&floats(&fx, "proba")).enumerate() {
        assert_close(p, q, &format!("logistic: proba[{i}]"));
    }
    let labels = Predictor::predict(&m, DenseView::new(&x, rows, cols).unwrap()).unwrap();
    let expected_labels: Vec<f64> = floats(&fx, "proba")
        .iter()
        .map(|&p| if p >= 0.5 { 1.0 } else { 0.0 })
        .collect();
    assert_eq!(labels, expected_labels);
}

/// The `elasticnet.kkt` contract must hold at the default tolerance, not only at the tight
/// tolerances the parity fixtures use. Sweeps penalties and mixing ratios on random data.
#[test]
fn elasticnet_kkt_contract_holds_at_default_tolerance() {
    let (rows, cols) = (50, 6);
    let mut seed: u64 = 12345;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let x: Vec<f64> = (0..rows * cols).map(|_| next()).collect();
    let y: Vec<f64> = (0..rows)
        .map(|r| x[r * cols] - 2.0 * x[r * cols + 3] + 0.3 * next())
        .collect();
    for &alpha in &[0.01, 0.1, 0.5] {
        for &rho in &[0.0, 0.3, 0.7, 1.0] {
            let mut m = tpt_ml_proven_linear::ElasticNet::<f64>::new(alpha, rho);
            SupervisedEstimator::fit(&mut m, DenseView::new(&x, rows, cols).unwrap(), &y[..])
                .unwrap_or_else(|e| panic!("alpha {alpha}, l1_ratio {rho}: {e}"));
        }
    }
}
