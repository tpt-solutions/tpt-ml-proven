// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Parity with scikit-learn's NMF (`init="custom"`, `solver="mu"`, Frobenius loss), on the
//! `nmf_*` fixtures from `tools/parity/generate_fixtures.py`. Both sides start from the same
//! factors and run the same multiplicative updates, so the iterates should agree closely.

use std::collections::HashMap;
use std::path::PathBuf;

use tpt_ml_proven_core::{ArrayView, DenseView, Transformer, UnsupervisedEstimator};
use tpt_ml_proven_decomposition::Nmf;

const TOL: f64 = 1e-9;

fn load(name: &str) -> HashMap<String, Vec<f64>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/parity/fixtures")
        .join(format!("{name}.txt"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|line| {
            let mut parts = line.split_whitespace();
            let key = parts.next().unwrap().to_string();
            (key, parts.filter_map(|s| s.parse().ok()).collect())
        })
        .collect()
}

fn assert_all_close(actual: &[f64], expected: &[f64], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let scale = 1.0 + e.abs();
        assert!(
            (a - e).abs() <= TOL * scale,
            "{what}[{i}]: {a} vs scikit-learn {e}"
        );
    }
}

fn check(name: &str) {
    let fx = load(name);
    let rows = fx["rows"][0] as usize;
    let cols = fx["cols"][0] as usize;
    let k = fx["k"][0] as usize;
    let max_iter = fx["max_iter"][0] as usize;
    let tol = fx["tol"][0];

    let mut model = Nmf::<f64>::new(k)
        .with_max_iter(max_iter)
        .with_tol(tol)
        .with_init(fx["init_w"].clone(), fx["init_h"].clone());
    UnsupervisedEstimator::fit(&mut model, DenseView::new(&fx["x"], rows, cols).unwrap()).unwrap();

    assert_eq!(
        model.n_iter().unwrap(),
        fx["n_iter"][0] as usize,
        "{name}: n_iter"
    );
    assert_all_close(model.components().unwrap(), &fx["h"], &format!("{name}: H"));

    let w = Transformer::fit_transform(&mut model, DenseView::new(&fx["x"], rows, cols).unwrap())
        .unwrap();
    let got: Vec<f64> = (0..rows * k).map(|i| w.view().get(i / k, i % k)).collect();
    assert_all_close(&got, &fx["w"], &format!("{name}: W"));
}

#[test]
fn nmf_fixed_iterations_match_scikit_learn() {
    check("nmf_mu_fixed_iters");
}

#[test]
fn nmf_with_tolerance_matches_scikit_learn() {
    check("nmf_mu_tol");
}
