// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Golden files and round trips for the random forests in the version 2 format. The golden
//! files are written by the ignored `regenerate_golden` test, never by hand.

use std::path::PathBuf;

use tpt_ml_proven_core::{DenseView, Predictor, SupervisedEstimator};
use tpt_ml_proven_trees::{RandomForestClassifier, RandomForestRegressor};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn inputs() -> Vec<f64> {
    (0..40)
        .flat_map(|i| [i as f64 * 0.1, ((i * 7) % 5) as f64])
        .collect()
}

fn classifier() -> RandomForestClassifier {
    let y: Vec<f64> = (0..40).map(|i| (i / 14) as f64).collect();
    let mut f = RandomForestClassifier::new(5, 3)
        .with_seed(11)
        .with_min_samples_leaf(2);
    SupervisedEstimator::fit(&mut f, DenseView::new(&inputs(), 40, 2).unwrap(), &y[..]).unwrap();
    f
}

fn regressor() -> RandomForestRegressor {
    let y: Vec<f64> = (0..40).map(|i| i as f64 * 0.3 - 1.0).collect();
    let mut f = RandomForestRegressor::new(4, 3)
        .with_seed(12)
        .with_max_features(1);
    SupervisedEstimator::fit(&mut f, DenseView::new(&inputs(), 40, 2).unwrap(), &y[..]).unwrap();
    f
}

#[test]
fn classifier_golden_file_loads_and_saves_identically() {
    let bytes =
        std::fs::read(fixture("random_forest_classifier_v2.tpml")).expect("golden file present");
    let f = RandomForestClassifier::from_bytes(&bytes).unwrap();
    assert_eq!(f.to_bytes().unwrap(), bytes);
}

#[test]
fn regressor_golden_file_loads_and_saves_identically() {
    let bytes =
        std::fs::read(fixture("random_forest_regressor_v2.tpml")).expect("golden file present");
    let f = RandomForestRegressor::from_bytes(&bytes).unwrap();
    assert_eq!(f.to_bytes().unwrap(), bytes);
}

#[test]
fn a_loaded_classifier_predicts_like_the_original() {
    let original = classifier();
    let loaded = RandomForestClassifier::from_bytes(&original.to_bytes().unwrap()).unwrap();
    assert_eq!(loaded, original);
    let x = inputs();
    let a = Predictor::predict(&original, DenseView::new(&x, 40, 2).unwrap()).unwrap();
    let b = Predictor::predict(&loaded, DenseView::new(&x, 40, 2).unwrap()).unwrap();
    assert_eq!(a, b);
}

#[test]
fn a_loaded_regressor_predicts_like_the_original() {
    let original = regressor();
    let loaded = RandomForestRegressor::from_bytes(&original.to_bytes().unwrap()).unwrap();
    assert_eq!(loaded, original);
    let x = inputs();
    let a = Predictor::predict(&original, DenseView::new(&x, 40, 2).unwrap()).unwrap();
    let b = Predictor::predict(&loaded, DenseView::new(&x, 40, 2).unwrap()).unwrap();
    assert_eq!(a, b);
}

#[test]
fn an_unfitted_forest_round_trips() {
    let c = RandomForestClassifier::new(3, 2);
    assert_eq!(
        RandomForestClassifier::from_bytes(&c.to_bytes().unwrap()).unwrap(),
        c
    );
    let r = RandomForestRegressor::new(3, 2).with_seed(4);
    assert_eq!(
        RandomForestRegressor::from_bytes(&r.to_bytes().unwrap()).unwrap(),
        r
    );
}

#[test]
fn a_classifier_file_is_not_a_regressor_file() {
    let bytes = classifier().to_bytes().unwrap();
    assert!(RandomForestRegressor::from_bytes(&bytes).is_err());
}

#[test]
#[ignore = "writes the golden files; run on purpose with --ignored"]
fn regenerate_golden() {
    std::fs::create_dir_all(fixture("")).unwrap();
    std::fs::write(
        fixture("random_forest_classifier_v2.tpml"),
        classifier().to_bytes().unwrap(),
    )
    .unwrap();
    std::fs::write(
        fixture("random_forest_regressor_v2.tpml"),
        regressor().to_bytes().unwrap(),
    )
    .unwrap();
}
