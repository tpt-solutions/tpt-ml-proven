//! Tests for the forest and boosting settings (split size, leaf limit, importances) and their
//! model files.

use tpt_ml_proven_core::{DenseView, Predictor, SupervisedEstimator};
use tpt_ml_proven_trees::{
    Criterion, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
};

/// Feature 0 decides three levels of y; feature 1 is constant.
fn steps() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..36).flat_map(|i| [i as f64, 2.0]).collect();
    let y: Vec<f64> = (0..36)
        .map(|i| match i / 12 {
            0 => 0.0,
            1 => 5.0,
            _ => 9.0,
        })
        .collect();
    let labels: Vec<f64> = y.iter().map(|v| v / 5.0).collect();
    (x, y, labels)
}

fn kind_of(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes[12..16].try_into().unwrap())
}

#[test]
fn a_forest_regressor_reports_importances_that_ignore_the_constant_feature() {
    let (x, y, _) = steps();
    let mut f = RandomForestRegressor::new(15, 4).with_seed(3);
    f.fit(DenseView::new(&x, 36, 2).unwrap(), &y[..]).unwrap();
    let imp = f.feature_importances().unwrap();
    assert!((imp.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert_eq!(imp, vec![1.0, 0.0]);
}

#[test]
fn a_forest_classifier_reports_importances_and_keeps_its_leaf_limit() {
    let (x, _, labels) = steps();
    let mut f = RandomForestClassifier::new(10, 6)
        .with_seed(2)
        .with_max_features(2)
        .with_min_samples_split(4)
        .with_max_leaf_nodes(Some(3));
    f.fit(DenseView::new(&x, 36, 2).unwrap(), &labels[..])
        .unwrap();
    // Each split considers one of the two features, so a tree whose root draws the constant
    // feature never splits and adds zeros. The average therefore sums to at most one.
    let imp = f.feature_importances().unwrap();
    assert!(imp.iter().sum::<f64>() <= 1.0 + 1e-12);
    assert_eq!(imp[1], 0.0);
    assert!(imp[0] > 0.0);
    assert_eq!(f.max_leaf_nodes(), Some(3));
    assert_eq!(f.min_samples_split(), 4);
    // Three labels in three bands; each tree has three leaves, so the vote should be accurate.
    let acc = f
        .score(DenseView::new(&x, 36, 2).unwrap(), &labels)
        .unwrap();
    assert!(acc >= 0.9, "accuracy {acc}");
}

#[test]
fn a_boosted_model_reports_importances_and_respects_the_leaf_limit() {
    let (x, y, _) = steps();
    let mut g = GradientBoostingRegressor::new(20, 4)
        .with_learning_rate(0.5)
        .with_max_leaf_nodes(Some(3));
    g.fit(DenseView::new(&x, 36, 2).unwrap(), &y[..]).unwrap();
    let imp = g.feature_importances().unwrap();
    assert!((imp.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert_eq!(imp, vec![1.0, 0.0]);
    assert_eq!(g.max_leaf_nodes(), Some(3));
}

#[test]
fn a_forest_with_new_settings_round_trips_in_the_new_kind() {
    let (x, _, labels) = steps();
    let mut f = RandomForestClassifier::new(6, 4)
        .with_seed(5)
        .with_criterion(Criterion::Entropy)
        .with_min_samples_split(3)
        .with_max_leaf_nodes(Some(4));
    f.fit(DenseView::new(&x, 36, 2).unwrap(), &labels[..])
        .unwrap();
    let bytes = f.to_bytes().unwrap();
    assert_eq!(kind_of(&bytes), 29);
    let back = RandomForestClassifier::from_bytes(&bytes).unwrap();
    assert_eq!(back, f);
    assert_eq!(back.feature_importances(), f.feature_importances());
}

#[test]
fn a_forest_regressor_round_trips_in_the_new_kind() {
    let (x, y, _) = steps();
    let mut f = RandomForestRegressor::new(5, 3)
        .with_seed(1)
        .with_max_leaf_nodes(Some(3));
    f.fit(DenseView::new(&x, 36, 2).unwrap(), &y[..]).unwrap();
    let bytes = f.to_bytes().unwrap();
    assert_eq!(kind_of(&bytes), 28);
    let back = RandomForestRegressor::from_bytes(&bytes).unwrap();
    assert_eq!(back, f);
    assert_eq!(
        back.predict(DenseView::new(&x, 36, 2).unwrap()).unwrap(),
        f.predict(DenseView::new(&x, 36, 2).unwrap()).unwrap()
    );
}

#[test]
fn a_boosted_model_with_new_settings_round_trips_in_the_new_kind() {
    let (x, y, _) = steps();
    let mut g = GradientBoostingRegressor::new(8, 3).with_max_leaf_nodes(Some(3));
    g.fit(DenseView::new(&x, 36, 2).unwrap(), &y[..]).unwrap();
    let bytes = g.to_bytes().unwrap();
    assert_eq!(kind_of(&bytes), 30);
    let back = GradientBoostingRegressor::from_bytes(&bytes).unwrap();
    assert_eq!(back, g);
    assert_eq!(back.feature_importances(), g.feature_importances());
}

#[test]
fn an_unfitted_forest_has_no_importances() {
    assert_eq!(RandomForestRegressor::new(3, 2).feature_importances(), None);
    assert_eq!(
        RandomForestClassifier::new(3, 2).feature_importances(),
        None
    );
    assert_eq!(
        GradientBoostingRegressor::new(3, 2).feature_importances(),
        None
    );
}

#[test]
fn invalid_forest_settings_are_rejected() {
    let (x, y, labels) = steps();
    let mut f = RandomForestRegressor::new(3, 2).with_min_samples_split(1);
    assert!(f.fit(DenseView::new(&x, 36, 2).unwrap(), &y[..]).is_err());
    let mut c = RandomForestClassifier::new(3, 2).with_max_leaf_nodes(Some(1));
    assert!(c
        .fit(DenseView::new(&x, 36, 2).unwrap(), &labels[..])
        .is_err());
    let mut g = GradientBoostingRegressor::new(3, 2).with_min_samples_split(0);
    assert!(g.fit(DenseView::new(&x, 36, 2).unwrap(), &y[..]).is_err());
}
