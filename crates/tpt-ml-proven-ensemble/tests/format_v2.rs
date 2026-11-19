// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Golden file for the version 2 format: a loaded forest must save back to the same bytes.
//! The file is written by the ignored `regenerate_golden` test, never by hand.

use std::path::PathBuf;

use tpt_ml_proven_core::{DenseView, SupervisedEstimator};
use tpt_ml_proven_ensemble::StumpForest;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn model() -> StumpForest<f64> {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..40 {
        let a = i as f64 / 10.0;
        x.extend([a, (i * 7 % 13) as f64]);
        y.push(if a < 2.0 { 0.0 } else { 1.0 });
    }
    let mut f = StumpForest::<f64>::new(9).with_seed(7);
    SupervisedEstimator::fit(&mut f, DenseView::new(&x, 40, 2).unwrap(), &y[..]).unwrap();
    f
}

#[test]
fn forest_golden_file_loads_and_saves_identically() {
    let bytes = std::fs::read(fixture("stump_forest_v2.tpml")).expect("golden file present");
    let f = StumpForest::<f64>::from_bytes(&bytes).unwrap();
    assert_eq!(f.to_bytes().unwrap(), bytes);
}

#[test]
#[ignore = "writes the golden file; run on purpose with --ignored"]
fn regenerate_golden() {
    std::fs::create_dir_all(fixture("")).unwrap();
    std::fs::write(fixture("stump_forest_v2.tpml"), model().to_bytes().unwrap()).unwrap();
}
