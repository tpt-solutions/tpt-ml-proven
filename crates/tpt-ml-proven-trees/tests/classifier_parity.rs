// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Parity with scikit-learn's `DecisionTreeClassifier` (Gini and entropy, exact splits) on the
//! `tree_classifier_*` fixtures. Predictions must match label for label, and the fitted tree
//! must have the same number of leaves and depth.

use std::collections::HashMap;
use std::path::PathBuf;

use tpt_ml_proven_core::{DenseView, Predictor, SupervisedEstimator};
use tpt_ml_proven_trees::{Criterion, DecisionTreeClassifier};

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

fn check(name: &str, criterion: Criterion) {
    let fx = load(name);
    let rows: usize = fx["rows"][0].parse().unwrap();
    let cols: usize = fx["cols"][0].parse().unwrap();
    let max_depth: usize = fx["max_depth"][0].parse().unwrap();
    let min_leaf: usize = fx["min_samples_leaf"][0].parse().unwrap();
    let x = floats(&fx, "x");
    let y = floats(&fx, "y");

    let mut model = DecisionTreeClassifier::new(max_depth)
        .with_min_samples_leaf(min_leaf)
        .with_criterion(criterion);
    SupervisedEstimator::fit(&mut model, DenseView::new(&x, rows, cols).unwrap(), &y[..]).unwrap();

    let pred = Predictor::predict(&model, DenseView::new(&x, rows, cols).unwrap()).unwrap();
    assert_eq!(pred, floats(&fx, "pred"), "{name}: predicted labels");

    let n_leaves: usize = fx["n_leaves"][0].parse().unwrap();
    let depth: usize = fx["depth"][0].parse().unwrap();
    assert_eq!(model.n_leaves(), Some(n_leaves), "{name}: leaves");
    assert_eq!(model.depth(), Some(depth), "{name}: depth");
}

#[test]
fn gini_tree_matches_scikit_learn() {
    check("tree_classifier_gini", Criterion::Gini);
}

#[test]
fn entropy_tree_matches_scikit_learn() {
    check("tree_classifier_entropy", Criterion::Entropy);
}
