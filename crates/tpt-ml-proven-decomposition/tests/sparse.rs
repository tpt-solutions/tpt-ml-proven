// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Property tests: a sparse (CSR) matrix and the same matrix stored densely must give the
//! same truncated SVD, because the sparse paths only skip exact zeros.

use tpt_ml_proven_core::{
    ArrayView, CsrView, Dense, DenseView, Transformer, UnsupervisedEstimator,
};
use tpt_ml_proven_decomposition::TruncatedSvd;

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

/// A random `rows × cols` matrix with roughly `density` of its entries non-zero, as dense
/// row-major data plus its CSR form.
fn random_matrix(
    seed: u64,
    rows: usize,
    cols: usize,
    density: f64,
) -> (Vec<f64>, Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut rng = Lcg(seed);
    let mut dense = vec![0.0; rows * cols];
    for v in dense.iter_mut() {
        if rng.unit() < density {
            *v = rng.unit() * 4.0 - 2.0;
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
fn sparse_and_dense_svd_agree_on_random_matrices() {
    for seed in 1..=8 {
        let (rows, cols) = (12 + seed as usize, 6);
        let density = 0.25 + 0.05 * seed as f64;
        let (dense, indptr, indices, data) = random_matrix(seed, rows, cols, density);

        let dense_view = DenseView::new(&dense, rows, cols).unwrap();
        let csr = CsrView::new(&indptr, &indices, &data, rows, cols).unwrap();

        let mut a = TruncatedSvd::<f64>::new(3);
        let mut b = TruncatedSvd::<f64>::new(3);
        UnsupervisedEstimator::fit(&mut a, dense_view).unwrap();
        UnsupervisedEstimator::fit(&mut b, csr).unwrap();

        for (sa, sb) in a
            .singular_values()
            .unwrap()
            .iter()
            .zip(b.singular_values().unwrap())
        {
            assert!(
                (sa - sb).abs() <= 1e-12 * sa.max(1.0),
                "seed {seed}: {sa} vs {sb}"
            );
        }
        let ca = a.components().unwrap();
        let cb = b.components().unwrap();
        for r in 0..3 {
            for c in 0..cols {
                assert!((ca.get(r, c) - cb.get(r, c)).abs() <= 1e-12, "seed {seed}");
            }
        }

        let za: Dense<f64> = a
            .transform(DenseView::new(&dense, rows, cols).unwrap())
            .unwrap();
        let zb: Dense<f64> = b
            .transform(CsrView::new(&indptr, &indices, &data, rows, cols).unwrap())
            .unwrap();
        for r in 0..rows {
            for c in 0..3 {
                assert!((za.get(r, c) - zb.get(r, c)).abs() <= 1e-12, "seed {seed}");
            }
        }
    }
}

#[test]
fn for_each_nonzero_visits_the_same_entries_on_both_layouts() {
    let (rows, cols) = (9, 7);
    let (dense, indptr, indices, data) = random_matrix(42, rows, cols, 0.3);
    let dense_view = DenseView::new(&dense, rows, cols).unwrap();
    let csr = CsrView::new(&indptr, &indices, &data, rows, cols).unwrap();
    for r in 0..rows {
        let mut from_dense = Vec::new();
        dense_view.for_each_nonzero(r, |c, v| from_dense.push((c, v)));
        let mut from_csr = Vec::new();
        csr.for_each_nonzero(r, |c, v| from_csr.push((c, v)));
        assert_eq!(from_dense, from_csr, "row {r}");
    }
}

#[test]
fn a_nan_stored_in_a_sparse_matrix_is_still_refused() {
    let indptr = [0, 1, 1];
    let indices = [0];
    let data = [f64::NAN];
    let csr = CsrView::new(&indptr, &indices, &data, 2, 2).unwrap();
    assert!(!csr.all_finite());
    // An empty sparse matrix has only implicit zeros, which are finite.
    let empty = CsrView::new(&[0, 0, 0], &[], &[] as &[f64], 2, 2).unwrap();
    assert!(empty.all_finite());
}
