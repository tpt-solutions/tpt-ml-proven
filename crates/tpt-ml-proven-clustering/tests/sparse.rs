// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Property tests: DBSCAN and HDBSCAN must give the same labels for a sparse (CSR) matrix as for
//! the same matrix stored densely. The sparse paths measure distances over the non-zeros only.

use tpt_ml_proven_clustering::{Dbscan, Hdbscan};
use tpt_ml_proven_core::{CsrView, DenseView, UnsupervisedEstimator};

/// A small deterministic generator, so the cases are reproducible.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }

    fn unit(&mut self) -> f64 {
        self.next() as f64 / (1u64 << 53) as f64
    }
}

/// Two groups of rows: the first near the origin on the first columns, the second near 5 on the
/// last columns, with roughly `density` of each row's entries non-zero. Returns the dense data
/// and its CSR parts.
fn grouped(
    seed: u64,
    rows: usize,
    cols: usize,
    density: f64,
) -> (Vec<f64>, Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut rng = Lcg(seed);
    let mut dense = vec![0.0; rows * cols];
    for r in 0..rows {
        let group_b = r >= rows / 2;
        for c in 0..cols {
            let in_group = if group_b { c >= cols / 2 } else { c < cols / 2 };
            if in_group && rng.unit() < density {
                let base = if group_b { 5.0 } else { 0.0 };
                dense[r * cols + c] = base + rng.unit();
            }
        }
    }
    let mut indptr = vec![0];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let v = dense[r * cols + c];
            if v != 0.0 {
                indices.push(c);
                data.push(v);
            }
        }
        indptr.push(indices.len());
    }
    (dense, indptr, indices, data)
}

#[test]
fn dbscan_sparse_labels_match_dense() {
    let (rows, cols) = (40, 6);
    let (dense, indptr, indices, data) = grouped(3, rows, cols, 0.7);
    let csr = CsrView::new(&indptr, &indices, &data, rows, cols).unwrap();
    let mut sparse_model = Dbscan::<f64>::new(2.0, 3);
    UnsupervisedEstimator::fit(&mut sparse_model, csr).unwrap();
    let mut dense_model = Dbscan::<f64>::new(2.0, 3);
    UnsupervisedEstimator::fit(
        &mut dense_model,
        DenseView::new(&dense, rows, cols).unwrap(),
    )
    .unwrap();

    assert_eq!(sparse_model.labels(), dense_model.labels());
    assert_eq!(
        sparse_model.core_sample_indices(),
        dense_model.core_sample_indices()
    );
    // The comparison is only meaningful if the data really forms clusters.
    assert!(dense_model.n_clusters().unwrap() >= 2);
}

#[test]
fn hdbscan_sparse_labels_match_dense() {
    let (rows, cols) = (40, 6);
    let (dense, indptr, indices, data) = grouped(9, rows, cols, 0.7);
    let csr = CsrView::new(&indptr, &indices, &data, rows, cols).unwrap();
    let mut sparse_model = Hdbscan::<f64>::new(4);
    UnsupervisedEstimator::fit(&mut sparse_model, csr).unwrap();
    let mut dense_model = Hdbscan::<f64>::new(4);
    UnsupervisedEstimator::fit(
        &mut dense_model,
        DenseView::new(&dense, rows, cols).unwrap(),
    )
    .unwrap();

    assert_eq!(sparse_model.labels(), dense_model.labels());
    assert!(dense_model.n_clusters().unwrap() >= 2);
}
