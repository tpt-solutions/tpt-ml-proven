// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Parity with scikit-learn's HDBSCAN (`cluster_selection_method="eom"`, `min_samples`
//! defaulting to `min_cluster_size`), on the `hdbscan_blobs_noise` fixture.
//!
//! Cluster numbering is not part of the comparison, so the check is that the two labelings
//! induce the same partition with the same noise points.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use tpt_ml_proven_clustering::hdbscan::NOISE;
use tpt_ml_proven_clustering::Hdbscan;
use tpt_ml_proven_core::{DenseView, UnsupervisedEstimator};

fn load(name: &str) -> HashMap<String, Vec<String>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/parity/fixtures")
        .join(format!("{name}.txt"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|line| {
            let mut parts = line.split_whitespace();
            let key = parts.next().unwrap().to_string();
            (key, parts.map(str::to_string).collect())
        })
        .collect()
}

#[test]
fn hdbscan_partition_matches_scikit_learn() {
    let fx = load("hdbscan_blobs_noise");
    let rows: usize = fx["rows"][0].parse().unwrap();
    let cols: usize = fx["cols"][0].parse().unwrap();
    let min_cluster_size: usize = fx["min_cluster_size"][0].parse().unwrap();
    let x: Vec<f64> = fx["x"].iter().map(|s| s.parse().unwrap()).collect();
    let expected: Vec<i64> = fx["labels"].iter().map(|s| s.parse().unwrap()).collect();

    let mut model = Hdbscan::<f64>::new(min_cluster_size);
    UnsupervisedEstimator::fit(&mut model, DenseView::new(&x, rows, cols).unwrap()).unwrap();
    let got = model.labels().unwrap();

    // Noise must agree exactly.
    for i in 0..rows {
        assert_eq!(
            got[i] == NOISE,
            expected[i] == NOISE,
            "noise mismatch at sample {i}"
        );
    }
    // Non-noise labels must agree up to renaming: a bijection between the two numberings.
    let mut forward: BTreeMap<i64, i64> = BTreeMap::new();
    let mut backward: BTreeMap<i64, i64> = BTreeMap::new();
    for i in 0..rows {
        if expected[i] == NOISE {
            continue;
        }
        let f = *forward.entry(got[i]).or_insert(expected[i]);
        let b = *backward.entry(expected[i]).or_insert(got[i]);
        assert_eq!(f, expected[i], "partition differs at sample {i}");
        assert_eq!(b, got[i], "partition differs at sample {i}");
    }
}
