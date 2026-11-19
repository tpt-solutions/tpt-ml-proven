// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Backward compatibility of the version 1 model format for PCA. The committed file in
//! `tests/fixtures` must keep loading. A failing test here means a format change broke existing
//! files: bump the format version rather than editing the fixture.
//!
//! To write the fixture (only when it is missing): `TPTML_WRITE_FIXTURES=1 cargo test -p
//! tpt-ml-proven-decomposition --test format_v1`.

use std::path::PathBuf;

use tpt_ml_proven_core::{DenseView, UnsupervisedEstimator};
use tpt_ml_proven_decomposition::Pca;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pca_v1.tpml")
}

fn data() -> Vec<f64> {
    vec![1.0, 2.0, 2.0, 4.1, 3.0, 5.9, 4.0, 8.2, 5.0, 9.9]
}

#[test]
fn version_one_pca_file_still_loads() {
    let x = data();
    if std::env::var_os("TPTML_WRITE_FIXTURES").is_some() {
        let mut p = Pca::<f64>::new(1);
        UnsupervisedEstimator::fit(&mut p, DenseView::new(&x, 5, 2).unwrap()).unwrap();
        std::fs::create_dir_all(fixture_path().parent().unwrap()).unwrap();
        std::fs::write(fixture_path(), p.to_bytes().unwrap()).unwrap();
    }

    let bytes = std::fs::read(fixture_path()).expect("fixture is committed");
    let loaded = Pca::<f64>::from_bytes(&bytes).expect("version 1 files must keep loading");

    let mut fresh = Pca::<f64>::new(1);
    UnsupervisedEstimator::fit(&mut fresh, DenseView::new(&x, 5, 2).unwrap()).unwrap();
    assert_eq!(loaded, fresh);
}
