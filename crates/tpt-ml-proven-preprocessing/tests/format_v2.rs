// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Golden files for the version 2 format: each loaded preprocessor must save back to the same
//! bytes. The files are written by the ignored `regenerate_golden` test, never by hand.

use std::path::PathBuf;

use tpt_ml_proven_core::{DenseView, UnsupervisedEstimator};
use tpt_ml_proven_preprocessing::{
    ImputeStrategy, MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, SimpleImputer,
    StandardScaler,
};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn scaler_data() -> Vec<f64> {
    vec![1.0, 10.0, 2.0, 30.0, 4.0, 10.0, 7.0, 50.0, 3.0, 20.0]
}

fn category_data() -> Vec<f64> {
    vec![3.0, 1.0, 1.0, 2.0, 3.0, 2.0]
}

fn imputer_data() -> Vec<f64> {
    vec![1.0, f64::NAN, 3.0, 4.0, 5.0, 4.0, f64::NAN, 2.0]
}

fn standard() -> StandardScaler<f64> {
    let mut m = StandardScaler::<f64>::new();
    UnsupervisedEstimator::fit(&mut m, DenseView::new(&scaler_data(), 5, 2).unwrap()).unwrap();
    m
}

fn minmax() -> MinMaxScaler<f64> {
    let mut m = MinMaxScaler::<f64>::new();
    UnsupervisedEstimator::fit(&mut m, DenseView::new(&scaler_data(), 5, 2).unwrap()).unwrap();
    m
}

fn robust() -> RobustScaler<f64> {
    let mut m = RobustScaler::<f64>::new();
    UnsupervisedEstimator::fit(&mut m, DenseView::new(&scaler_data(), 5, 2).unwrap()).unwrap();
    m
}

fn one_hot() -> OneHotEncoder<f64> {
    let mut m = OneHotEncoder::<f64>::new();
    UnsupervisedEstimator::fit(&mut m, DenseView::new(&category_data(), 3, 2).unwrap()).unwrap();
    m
}

fn ordinal() -> OrdinalEncoder<f64> {
    let mut m = OrdinalEncoder::<f64>::new();
    UnsupervisedEstimator::fit(&mut m, DenseView::new(&category_data(), 3, 2).unwrap()).unwrap();
    m
}

fn imputer() -> SimpleImputer<f64> {
    let mut m = SimpleImputer::<f64>::new(ImputeStrategy::Median);
    UnsupervisedEstimator::fit(&mut m, DenseView::new(&imputer_data(), 4, 2).unwrap()).unwrap();
    m
}

#[test]
fn standard_scaler_golden_file_is_stable() {
    let bytes = std::fs::read(fixture("standard_scaler_v2.tpml")).expect("golden file present");
    let m = StandardScaler::<f64>::from_bytes(&bytes).unwrap();
    assert_eq!(m.to_bytes().unwrap(), bytes);
}

#[test]
fn minmax_scaler_golden_file_is_stable() {
    let bytes = std::fs::read(fixture("minmax_scaler_v2.tpml")).expect("golden file present");
    let m = MinMaxScaler::<f64>::from_bytes(&bytes).unwrap();
    assert_eq!(m.to_bytes().unwrap(), bytes);
}

#[test]
fn robust_scaler_golden_file_is_stable() {
    let bytes = std::fs::read(fixture("robust_scaler_v2.tpml")).expect("golden file present");
    let m = RobustScaler::<f64>::from_bytes(&bytes).unwrap();
    assert_eq!(m.to_bytes().unwrap(), bytes);
}

#[test]
fn one_hot_golden_file_is_stable() {
    let bytes = std::fs::read(fixture("one_hot_v2.tpml")).expect("golden file present");
    let m = OneHotEncoder::<f64>::from_bytes(&bytes).unwrap();
    assert_eq!(m.to_bytes().unwrap(), bytes);
}

#[test]
fn ordinal_golden_file_is_stable() {
    let bytes = std::fs::read(fixture("ordinal_v2.tpml")).expect("golden file present");
    let m = OrdinalEncoder::<f64>::from_bytes(&bytes).unwrap();
    assert_eq!(m.to_bytes().unwrap(), bytes);
}

#[test]
fn imputer_golden_file_is_stable() {
    let bytes = std::fs::read(fixture("simple_imputer_v2.tpml")).expect("golden file present");
    let m = SimpleImputer::<f64>::from_bytes(&bytes).unwrap();
    assert_eq!(m.to_bytes().unwrap(), bytes);
}

#[test]
#[ignore = "writes the golden files; run on purpose with --ignored"]
fn regenerate_golden() {
    std::fs::create_dir_all(fixture("")).unwrap();
    let files = [
        ("standard_scaler_v2.tpml", standard().to_bytes().unwrap()),
        ("minmax_scaler_v2.tpml", minmax().to_bytes().unwrap()),
        ("robust_scaler_v2.tpml", robust().to_bytes().unwrap()),
        ("one_hot_v2.tpml", one_hot().to_bytes().unwrap()),
        ("ordinal_v2.tpml", ordinal().to_bytes().unwrap()),
        ("simple_imputer_v2.tpml", imputer().to_bytes().unwrap()),
    ];
    for (name, bytes) in files {
        std::fs::write(fixture(name), bytes).unwrap();
    }
}
