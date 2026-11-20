// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Property tests: a sparse (CSR) matrix and the same matrix stored densely must give the same
//! Ridge, ElasticNet and Lasso models. The sparse paths work from the non-zeros and never copy
//! the matrix into a dense buffer, so these tests are what keeps the two paths honest.

use tpt_ml_proven_core::{CsrView, DenseView, Predictor, SupervisedEstimator};
use tpt_ml_proven_linear::{ElasticNet, Lasso, Ridge};

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

/// A sparse test problem: dense row-major data, its CSR parts, and a target made from the data.
struct Problem {
    dense: Vec<f64>,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
    y: Vec<f64>,
    rows: usize,
    cols: usize,
}

/// A random `rows × cols` matrix with roughly `density` of its entries non-zero. Column
/// `empty_col`, when given, has no non-zeros at all. The target is a linear function of the
/// data plus a little noise.
fn problem(seed: u64, rows: usize, cols: usize, density: f64, empty_col: Option<usize>) -> Problem {
    let mut rng = Lcg(seed);
    let mut dense = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            if Some(c) != empty_col && rng.unit() < density {
                dense[r * cols + c] = rng.unit() * 4.0 - 2.0;
            }
        }
    }
    let weights: Vec<f64> = (0..cols).map(|c| (c as f64 + 1.0) * 0.5 - 1.0).collect();
    let y: Vec<f64> = (0..rows)
        .map(|r| {
            let signal: f64 = (0..cols).map(|c| dense[r * cols + c] * weights[c]).sum();
            signal + 0.05 * (rng.unit() - 0.5) + 2.0
        })
        .collect();
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
    Problem {
        dense,
        indptr,
        indices,
        data,
        y,
        rows,
        cols,
    }
}

fn assert_close(label: &str, a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "{label}: length");
    let scale = 1.0 + b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol * scale,
            "{label}[{i}]: sparse {x} vs dense {y}"
        );
    }
}

#[test]
fn ridge_sparse_agrees_with_dense() {
    for seed in 1..=4 {
        let p = problem(seed, 60, 8, 0.3, None);
        let csr = CsrView::new(&p.indptr, &p.indices, &p.data, p.rows, p.cols).unwrap();
        let dense = DenseView::new(&p.dense, p.rows, p.cols).unwrap();

        let mut sparse_model = Ridge::<f64>::new(0.5);
        SupervisedEstimator::fit(&mut sparse_model, csr, &p.y[..]).unwrap();
        let mut dense_model = Ridge::<f64>::new(0.5);
        SupervisedEstimator::fit(&mut dense_model, dense, &p.y[..]).unwrap();

        assert_close(
            "ridge coef",
            sparse_model.coef().unwrap(),
            dense_model.coef().unwrap(),
            1e-9,
        );
        let (a, b) = (
            sparse_model.intercept().unwrap(),
            dense_model.intercept().unwrap(),
        );
        assert!((a - b).abs() < 1e-9, "intercept {a} vs {b}");

        let csr = CsrView::new(&p.indptr, &p.indices, &p.data, p.rows, p.cols).unwrap();
        let dense = DenseView::new(&p.dense, p.rows, p.cols).unwrap();
        let ps = Predictor::predict(&sparse_model, csr).unwrap();
        let pd = Predictor::predict(&dense_model, dense).unwrap();
        assert_close("ridge predictions", &ps, &pd, 1e-9);
    }
}

#[test]
fn an_empty_column_gets_a_zero_coefficient_on_both_paths() {
    let p = problem(7, 40, 5, 0.4, Some(2));
    let csr = CsrView::new(&p.indptr, &p.indices, &p.data, p.rows, p.cols).unwrap();
    let mut sparse_model = Ridge::<f64>::new(1.0);
    SupervisedEstimator::fit(&mut sparse_model, csr, &p.y[..]).unwrap();
    let dense = DenseView::new(&p.dense, p.rows, p.cols).unwrap();
    let mut dense_model = Ridge::<f64>::new(1.0);
    SupervisedEstimator::fit(&mut dense_model, dense, &p.y[..]).unwrap();
    assert_eq!(sparse_model.coef().unwrap()[2], 0.0);
    assert_close(
        "ridge coef with an empty column",
        sparse_model.coef().unwrap(),
        dense_model.coef().unwrap(),
        1e-9,
    );
}

#[test]
fn elastic_net_sparse_agrees_with_dense() {
    for seed in 1..=3 {
        let p = problem(seed, 80, 6, 0.35, None);
        let csr = CsrView::new(&p.indptr, &p.indices, &p.data, p.rows, p.cols).unwrap();
        let dense = DenseView::new(&p.dense, p.rows, p.cols).unwrap();

        let mut sparse_model = ElasticNet::<f64>::new(0.05, 0.5)
            .with_max_iter(20000)
            .with_tol(1e-12);
        SupervisedEstimator::fit(&mut sparse_model, csr, &p.y[..]).unwrap();
        let mut dense_model = ElasticNet::<f64>::new(0.05, 0.5)
            .with_max_iter(20000)
            .with_tol(1e-12);
        SupervisedEstimator::fit(&mut dense_model, dense, &p.y[..]).unwrap();

        assert_close(
            "elastic net coef",
            sparse_model.coef().unwrap(),
            dense_model.coef().unwrap(),
            1e-6,
        );
        let (a, b) = (
            sparse_model.intercept().unwrap(),
            dense_model.intercept().unwrap(),
        );
        assert!((a - b).abs() < 1e-6, "intercept {a} vs {b}");
    }
}

#[test]
fn lasso_sparse_agrees_with_dense_including_the_zeros() {
    let p = problem(11, 80, 6, 0.35, None);
    let csr = CsrView::new(&p.indptr, &p.indices, &p.data, p.rows, p.cols).unwrap();
    let dense = DenseView::new(&p.dense, p.rows, p.cols).unwrap();

    let mut sparse_model = Lasso::<f64>::new(0.1).with_max_iter(20000).with_tol(1e-12);
    SupervisedEstimator::fit(&mut sparse_model, csr, &p.y[..]).unwrap();
    let mut dense_model = Lasso::<f64>::new(0.1).with_max_iter(20000).with_tol(1e-12);
    SupervisedEstimator::fit(&mut dense_model, dense, &p.y[..]).unwrap();

    let (sc, dc) = (sparse_model.coef().unwrap(), dense_model.coef().unwrap());
    assert_close("lasso coef", sc, dc, 1e-6);
    for (i, (a, b)) in sc.iter().zip(dc).enumerate() {
        assert_eq!(
            *a == 0.0,
            *b == 0.0,
            "coefficient {i} is zero on one path only"
        );
    }
}
