// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Golden file and round trip for the classification tree in the version 2 format. The golden
//! file is written by the ignored `regenerate_golden` test, never by hand.

use std::path::PathBuf;

use tpt_ml_proven_core::{DenseView, Predictor, SupervisedEstimator};
use tpt_ml_proven_trees::{Criterion, DecisionTreeClassifier};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn model() -> DecisionTreeClassifier {
    // Three classes along the first feature, with the second feature as noise.
    let x: Vec<f64> = (0..30)
        .flat_map(|i| [i as f64 * 0.1, ((i * 7) % 5) as f64])
        .collect();
    let y: Vec<f64> = (0..30).map(|i| (i / 10) as f64).collect();
    let mut t = DecisionTreeClassifier::new(3).with_criterion(Criterion::Entropy);
    SupervisedEstimator::fit(&mut t, DenseView::new(&x, 30, 2).unwrap(), &y[..]).unwrap();
    t
}

#[test]
fn classifier_golden_file_loads_and_saves_identically() {
    let bytes = std::fs::read(fixture("tree_classifier_v2.tpml")).expect("golden file present");
    let t = DecisionTreeClassifier::from_bytes(&bytes).unwrap();
    assert_eq!(t.to_bytes().unwrap(), bytes);
}

#[test]
fn a_loaded_classifier_predicts_like_the_original() {
    let original = model();
    let loaded = DecisionTreeClassifier::from_bytes(&original.to_bytes().unwrap()).unwrap();
    assert_eq!(loaded, original);
    let x: Vec<f64> = (0..30)
        .flat_map(|i| [i as f64 * 0.1, ((i * 7) % 5) as f64])
        .collect();
    let a = Predictor::predict(&original, DenseView::new(&x, 30, 2).unwrap()).unwrap();
    let b = Predictor::predict(&loaded, DenseView::new(&x, 30, 2).unwrap()).unwrap();
    assert_eq!(a, b);
}

#[test]
fn an_unfitted_classifier_round_trips() {
    let t = DecisionTreeClassifier::new(4).with_min_samples_leaf(3);
    assert_eq!(
        DecisionTreeClassifier::from_bytes(&t.to_bytes().unwrap()).unwrap(),
        t
    );
}

#[test]
#[ignore = "writes the golden file; run on purpose with --ignored"]
fn regenerate_golden() {
    std::fs::create_dir_all(fixture("")).unwrap();
    std::fs::write(
        fixture("tree_classifier_v2.tpml"),
        model().to_bytes().unwrap(),
    )
    .unwrap();
}
