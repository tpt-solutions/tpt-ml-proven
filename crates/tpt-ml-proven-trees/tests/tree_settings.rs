// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions
//
// The single-tree settings added with the version 2 tree files: the split size, the leaf limit
// and feature importances, and their model files.

//! Tests for the single-tree settings (split size, leaf limit, importances) and their model files.

use tpt_ml_proven_core::{DenseView, Predictor, SupervisedEstimator};
use tpt_ml_proven_trees::{Criterion, DecisionTreeClassifier, DecisionTreeRegressor};

/// A step function of feature 0 with three levels, and a constant second feature.
fn steps() -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..18).flat_map(|i| [i as f64, 2.0]).collect();
    let y: Vec<f64> = (0..18)
        .map(|i| match i / 6 {
            0 => 0.0,
            1 => 5.0,
            _ => 9.0,
        })
        .collect();
    (x, y)
}

#[test]
fn a_leaf_limit_is_respected_and_grown_best_first() {
    let (x, y) = steps();
    let mut t = DecisionTreeRegressor::new(10).with_max_leaf_nodes(Some(3));
    t.fit(DenseView::new(&x, 18, 2).unwrap(), &y[..]).unwrap();
    assert_eq!(t.tree().unwrap().n_leaves(), 3);
    // Three levels, three leaves: the limit still allows an exact fit.
    let pred = t.predict(DenseView::new(&x, 18, 2).unwrap()).unwrap();
    assert_eq!(pred, y);
}

#[test]
fn importances_are_shares_of_the_error_reduction() {
    let (x, y) = steps();
    let mut t = DecisionTreeRegressor::new(10);
    t.fit(DenseView::new(&x, 18, 2).unwrap(), &y[..]).unwrap();
    let imp = t.feature_importances().unwrap();
    assert!((imp.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    // The constant feature is never split on.
    assert_eq!(imp, vec![1.0, 0.0]);
}

#[test]
fn min_samples_split_stops_small_nodes_from_splitting() {
    let (x, y) = steps();
    let mut t = DecisionTreeRegressor::new(10).with_min_samples_split(19);
    t.fit(DenseView::new(&x, 18, 2).unwrap(), &y[..]).unwrap();
    assert_eq!(t.tree().unwrap().n_leaves(), 1);
}

#[test]
fn invalid_settings_are_rejected() {
    let (x, y) = steps();
    let view = || DenseView::new(&x, 18, 2).unwrap();
    let mut split = DecisionTreeRegressor::new(3).with_min_samples_split(1);
    assert!(split.fit(view(), &y[..]).is_err());
    let mut leaves = DecisionTreeRegressor::new(3).with_max_leaf_nodes(Some(1));
    assert!(leaves.fit(view(), &y[..]).is_err());
}

#[test]
fn a_regressor_with_new_settings_round_trips_in_the_new_kind() {
    let (x, y) = steps();
    let mut t = DecisionTreeRegressor::new(4)
        .with_min_samples_split(3)
        .with_max_leaf_nodes(Some(3));
    t.fit(DenseView::new(&x, 18, 2).unwrap(), &y[..]).unwrap();
    let bytes = t.to_bytes().unwrap();
    // Kind 26 is stored at byte 12, after the magic and the version.
    assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 26);
    let back = DecisionTreeRegressor::from_bytes(&bytes).unwrap();
    assert_eq!(back, t);
    assert_eq!(back.min_samples_split(), 3);
    assert_eq!(back.max_leaf_nodes(), Some(3));
    assert_eq!(back.feature_importances(), t.feature_importances());
}

#[test]
fn a_default_regressor_with_no_importances_keeps_the_older_kind() {
    // An unfitted model with default settings is representable in kind 8.
    let t = DecisionTreeRegressor::new(3);
    let bytes = t.to_bytes().unwrap();
    assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 8);
    let back = DecisionTreeRegressor::from_bytes(&bytes).unwrap();
    assert_eq!(back, t);
}

#[test]
fn a_classifier_with_new_settings_round_trips_in_the_new_kind() {
    let (x, y) = steps();
    let classes: Vec<f64> = y.iter().map(|v| v / 5.0).collect();
    let mut t = DecisionTreeClassifier::new(4)
        .with_criterion(Criterion::Entropy)
        .with_min_samples_split(3)
        .with_max_leaf_nodes(Some(3));
    t.fit(DenseView::new(&x, 18, 2).unwrap(), &classes[..])
        .unwrap();
    let bytes = t.to_bytes().unwrap();
    assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 27);
    let back = DecisionTreeClassifier::from_bytes(&bytes).unwrap();
    assert_eq!(back, t);
    assert_eq!(back.max_leaf_nodes(), Some(3));
    assert_eq!(back.feature_importances(), t.feature_importances());
    assert_eq!(
        back.predict(DenseView::new(&x, 18, 2).unwrap()).unwrap(),
        t.predict(DenseView::new(&x, 18, 2).unwrap()).unwrap()
    );
}

#[test]
fn a_truncated_model_file_is_refused() {
    let (x, y) = steps();
    let mut t = DecisionTreeRegressor::new(4);
    t.fit(DenseView::new(&x, 18, 2).unwrap(), &y[..]).unwrap();
    let mut bytes = t.to_bytes().unwrap();
    assert!(DecisionTreeRegressor::from_bytes(&bytes).is_ok());
    bytes.truncate(bytes.len() - 4);
    assert!(DecisionTreeRegressor::from_bytes(&bytes).is_err());
}
