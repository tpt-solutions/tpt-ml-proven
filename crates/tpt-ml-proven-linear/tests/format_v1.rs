// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Backward compatibility of the version 1 model format. The file in `tests/fixtures` was
//! written once by this crate and is committed. If a later change makes this test fail, the
//! change breaks existing model files and needs a new format version, not an edit here.
//!
//! To regenerate the fixture (only when adding a new fixture, never to make a failing test
//! pass): `TPTML_WRITE_FIXTURES=1 cargo test -p tpt-ml-proven-linear --test format_v1`.

use std::path::PathBuf;

use tpt_ml_proven_core::{DenseView, Predictor, SupervisedEstimator};
use tpt_ml_proven_linear::Ridge;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ridge_v1.tpml")
}

fn training_data() -> (Vec<f64>, Vec<f64>) {
    let x = vec![0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0, 2.0, 3.0, 1.0, 0.0, 3.0];
    let y = vec![-1.0, 1.0, 3.5, 1.0, 1.0, -0.5];
    (x, y)
}

#[test]
fn version_one_ridge_file_still_loads_and_predicts() {
    let (x, y) = training_data();
    if std::env::var_os("TPTML_WRITE_FIXTURES").is_some() {
        let mut m = Ridge::<f64>::new(0.5);
        SupervisedEstimator::fit(&mut m, DenseView::new(&x, 6, 2).unwrap(), &y[..]).unwrap();
        std::fs::create_dir_all(fixture_path().parent().unwrap()).unwrap();
        std::fs::write(fixture_path(), m.to_bytes().unwrap()).unwrap();
    }

    let bytes = std::fs::read(fixture_path()).expect("fixture is committed");
    let loaded = Ridge::<f64>::from_bytes(&bytes).expect("version 1 files must keep loading");

    // The loaded model must reproduce a fresh fit with the same parameter.
    let mut fresh = Ridge::<f64>::new(0.5);
    SupervisedEstimator::fit(&mut fresh, DenseView::new(&x, 6, 2).unwrap(), &y[..]).unwrap();
    assert_eq!(loaded, fresh);

    let p = Predictor::predict(&loaded, DenseView::new(&x, 6, 2).unwrap()).unwrap();
    assert_eq!(p.len(), 6);
}

#[test]
fn version_one_header_is_stable() {
    let bytes = std::fs::read(fixture_path()).expect("fixture is committed");
    assert_eq!(&bytes[..8], b"TPTMLMDL");
    assert_eq!(&bytes[8..12], &1_u32.to_le_bytes());
}
