// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Backward compatibility of the version 1 model format for KMeans. The committed file in
//! `tests/fixtures` must keep loading. A failing test here means a format change broke existing
//! files: bump the format version rather than editing the fixture.
//!
//! To write the fixture (only when it is missing): `TPTML_WRITE_FIXTURES=1 cargo test -p
//! tpt-ml-proven-clustering --test format_v1`.

use std::path::PathBuf;

use tpt_ml_proven_clustering::KMeans;
use tpt_ml_proven_core::{DenseView, UnsupervisedEstimator};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kmeans_v1.tpml")
}

fn data() -> Vec<f64> {
    vec![
        0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
    ]
}

#[test]
fn version_one_kmeans_file_still_loads() {
    let x = data();
    if std::env::var_os("TPTML_WRITE_FIXTURES").is_some() {
        let mut km = KMeans::<f64>::new(2).with_seed(3);
        UnsupervisedEstimator::fit(&mut km, DenseView::new(&x, 6, 2).unwrap()).unwrap();
        std::fs::create_dir_all(fixture_path().parent().unwrap()).unwrap();
        std::fs::write(fixture_path(), km.to_bytes().unwrap()).unwrap();
    }

    let bytes = std::fs::read(fixture_path()).expect("fixture is committed");
    let loaded = KMeans::<f64>::from_bytes(&bytes).expect("version 1 files must keep loading");

    let mut fresh = KMeans::<f64>::new(2).with_seed(3);
    UnsupervisedEstimator::fit(&mut fresh, DenseView::new(&x, 6, 2).unwrap()).unwrap();
    assert_eq!(loaded, fresh);
}
