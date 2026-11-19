// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! The sampling options of gradient boosting: row subsampling, per-split feature limits and
//! the seed, plus loading files written before those options existed.

use tpt_ml_proven_core::persist::{dtype, kind, Writer};
use tpt_ml_proven_core::{DenseView, Error, Predictor, SupervisedEstimator};
use tpt_ml_proven_trees::GradientBoostingRegressor;

/// y = 2·x0 − x1 + 0.5·x2 on a deterministic grid, plus a small wobble.
fn data() -> (Vec<f64>, Vec<f64>, usize) {
    let rows = 80;
    let mut x = Vec::with_capacity(rows * 3);
    let mut y = Vec::with_capacity(rows);
    for i in 0..rows {
        let a = (i as f64 * 0.37).sin() * 3.0;
        let b = (i as f64 * 0.11).cos() * 2.0;
        let c = (i % 9) as f64 - 4.0;
        x.extend([a, b, c]);
        y.push(2.0 * a - b + 0.5 * c + ((i * 7) % 5) as f64 * 0.02);
    }
    (x, y, rows)
}

fn fit(model: &mut GradientBoostingRegressor) {
    let (x, y, rows) = data();
    SupervisedEstimator::fit(model, DenseView::new(&x, rows, 3).unwrap(), &y[..]).unwrap();
}

fn predictions(model: &GradientBoostingRegressor) -> Vec<f64> {
    let (x, _, rows) = data();
    Predictor::predict(model, DenseView::new(&x, rows, 3).unwrap()).unwrap()
}

#[test]
fn subsampled_boosting_still_learns_the_signal() {
    let (_, y, _) = data();
    let mut m = GradientBoostingRegressor::new(60, 3)
        .with_learning_rate(0.2)
        .with_subsample(0.6)
        .with_seed(4);
    fit(&mut m);
    let pred = predictions(&m);
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    let ss_tot: f64 = y.iter().map(|v| (v - mean) * (v - mean)).sum();
    let ss_res: f64 = y.iter().zip(&pred).map(|(t, p)| (t - p) * (t - p)).sum();
    assert!(1.0 - ss_res / ss_tot > 0.9, "subsampled R² too low");
}

#[test]
fn the_same_seed_gives_the_same_model_and_another_seed_a_different_one() {
    let build = |seed| {
        let mut m = GradientBoostingRegressor::new(10, 2)
            .with_subsample(0.5)
            .with_max_features(2)
            .with_seed(seed);
        fit(&mut m);
        m
    };
    assert_eq!(build(7), build(7));
    assert_ne!(build(7), build(8));
}

#[test]
fn feature_limited_boosting_is_seeded_and_fits() {
    let mut m = GradientBoostingRegressor::new(30, 3)
        .with_max_features(1)
        .with_seed(2);
    fit(&mut m);
    assert!(predictions(&m).iter().all(|p| p.is_finite()));
    let mut again = GradientBoostingRegressor::new(30, 3)
        .with_max_features(1)
        .with_seed(2);
    fit(&mut again);
    assert_eq!(m, again);
}

#[test]
fn invalid_sampling_settings_are_refused() {
    let (x, y, rows) = data();
    for bad in [0.0, -0.5, 1.5, f64::NAN] {
        let mut m = GradientBoostingRegressor::new(3, 2).with_subsample(bad);
        assert!(matches!(
            SupervisedEstimator::fit(&mut m, DenseView::new(&x, rows, 3).unwrap(), &y[..]),
            Err(Error::InvalidParameter {
                name: "subsample",
                ..
            })
        ));
    }
    let mut zero = GradientBoostingRegressor::new(3, 2).with_max_features(0);
    assert!(matches!(
        SupervisedEstimator::fit(&mut zero, DenseView::new(&x, rows, 3).unwrap(), &y[..]),
        Err(Error::InvalidParameter {
            name: "max_features",
            ..
        })
    ));
}

#[test]
fn sampled_model_round_trips_with_its_settings() {
    let mut m = GradientBoostingRegressor::new(8, 2)
        .with_subsample(0.7)
        .with_max_features(2)
        .with_seed(31);
    fit(&mut m);
    let loaded = GradientBoostingRegressor::from_bytes(&m.to_bytes().unwrap()).unwrap();
    assert_eq!(loaded, m);
    assert_eq!(predictions(&loaded), predictions(&m));
}

#[test]
fn a_kind_nine_file_loads_with_the_default_sampling_settings() {
    // The layout written before the sampling settings existed: no subsample, feature or seed fields.
    let mut w = Writer::with_dtype(kind::GRADIENT_BOOSTING, dtype::F64);
    w.usize(5);
    w.f64(0.1);
    w.usize(2);
    w.usize(1);
    w.bool(false);
    let loaded = GradientBoostingRegressor::from_bytes(&w.finish()).unwrap();
    assert_eq!(
        loaded,
        GradientBoostingRegressor::new(5, 2).with_learning_rate(0.1)
    );
}
