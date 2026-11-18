// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Parity with scikit-learn's StandardScaler, MinMaxScaler and RobustScaler, on the
//! `scalers_skewed` fixture from `tools/parity/generate_fixtures.py`.

use std::collections::HashMap;
use std::path::PathBuf;

use tpt_ml_proven_core::{ArrayView, Dense, DenseView, Transformer, UnsupervisedEstimator};
use tpt_ml_proven_preprocessing::{MinMaxScaler, RobustScaler, StandardScaler};

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
            let values = parts.filter_map(|s| s.parse().ok()).collect();
            (key, values)
        })
        .collect()
}

/// Row-major values of a transformed matrix.
fn flat(m: &Dense<f64>) -> Vec<f64> {
    let v = m.view();
    (0..v.rows())
        .flat_map(|r| (0..v.cols()).map(move |c| (r, c)))
        .map(|(r, c)| v.get(r, c))
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

#[test]
fn scalers_match_scikit_learn() {
    let fx = load("scalers_skewed");
    let rows = fx["rows"][0] as usize;
    let cols = fx["cols"][0] as usize;
    let x = &fx["x"];
    let view = || DenseView::new(x.as_slice(), rows, cols).unwrap();

    let mut standard = StandardScaler::<f64>::new();
    UnsupervisedEstimator::fit(&mut standard, view()).unwrap();
    assert_all_close(
        standard.mean().unwrap(),
        &fx["standard_mean"],
        "standard mean",
    );
    assert_all_close(
        standard.scale().unwrap(),
        &fx["standard_scale"],
        "standard scale",
    );
    let out = Transformer::transform(&standard, view()).unwrap();
    assert_all_close(&flat(&out), &fx["standard_out"], "standard output");

    let mut minmax = MinMaxScaler::<f64>::new();
    UnsupervisedEstimator::fit(&mut minmax, view()).unwrap();
    let out = Transformer::transform(&minmax, view()).unwrap();
    assert_all_close(&flat(&out), &fx["minmax_out"], "minmax output");

    let mut robust = RobustScaler::<f64>::new();
    UnsupervisedEstimator::fit(&mut robust, view()).unwrap();
    let out = Transformer::transform(&robust, view()).unwrap();
    assert_all_close(&flat(&out), &fx["robust_out"], "robust output");
}
