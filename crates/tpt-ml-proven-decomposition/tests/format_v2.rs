// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Golden files for the version 2 format: a loaded model must save back to the same bytes,
//! which pins the field layout. The files are written by the ignored `regenerate_golden`
//! test, never by hand.

use std::path::PathBuf;

use tpt_ml_proven_core::{DenseView, UnsupervisedEstimator};
use tpt_ml_proven_decomposition::{Nmf, TruncatedSvd};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn svd_data() -> Vec<f64> {
    (0..30).map(|i| ((i * 7) % 13) as f64 - 6.0).collect()
}

fn nmf_data() -> Vec<f64> {
    (0..12).map(|i| ((i * 5) % 7) as f64 + 0.5).collect()
}

fn svd_model() -> TruncatedSvd<f64> {
    let mut m = TruncatedSvd::<f64>::new(2);
    UnsupervisedEstimator::fit(&mut m, DenseView::new(&svd_data(), 10, 3).unwrap()).unwrap();
    m
}

fn nmf_model() -> Nmf<f64> {
    let mut m = Nmf::<f64>::new(2).with_seed(3);
    UnsupervisedEstimator::fit(&mut m, DenseView::new(&nmf_data(), 4, 3).unwrap()).unwrap();
    m
}

#[test]
fn svd_golden_file_loads_and_saves_identically() {
    let bytes = std::fs::read(fixture("svd_v2.tpml")).expect("golden file present");
    let m = TruncatedSvd::<f64>::from_bytes(&bytes).unwrap();
    assert_eq!(m.to_bytes().unwrap(), bytes);
}

#[test]
fn nmf_golden_file_loads_and_saves_identically() {
    let bytes = std::fs::read(fixture("nmf_v2.tpml")).expect("golden file present");
    let m = Nmf::<f64>::from_bytes(&bytes).unwrap();
    assert_eq!(m.to_bytes().unwrap(), bytes);
}

#[test]
#[ignore = "writes the golden files; run on purpose with --ignored"]
fn regenerate_golden() {
    std::fs::create_dir_all(fixture("")).unwrap();
    std::fs::write(fixture("svd_v2.tpml"), svd_model().to_bytes().unwrap()).unwrap();
    std::fs::write(fixture("nmf_v2.tpml"), nmf_model().to_bytes().unwrap()).unwrap();
}
