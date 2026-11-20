// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Times KMeans on the same sparse matrix stored as CSR and as dense, single-threaded, and
//! reports the memory each form needs.
//!
//! Run with `cargo run --release -p tpt-ml-proven-clustering --example bench_sparse_kmeans --
//! [rows] [cols] [density] [k] [iters]`. The defaults are 100000 × 200 at 5% density, k = 16.

use std::time::Instant;

use tpt_ml_proven_clustering::KMeans;
use tpt_ml_proven_core::{CsrView, DenseView, UnsupervisedEstimator};

/// A small deterministic generator.
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

fn arg<T: std::str::FromStr>(args: &[String], i: usize, default: T) -> T {
    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rows: usize = arg(&args, 0, 100_000);
    let cols: usize = arg(&args, 1, 200);
    let density: f64 = arg(&args, 2, 0.05);
    let k: usize = arg(&args, 3, 16);
    let iters: usize = arg(&args, 4, 20);

    let mut rng = Lcg(7);
    let mut dense = vec![0.0_f64; rows * cols];
    for v in dense.iter_mut() {
        if rng.unit() < density {
            *v = rng.unit();
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

    let fit_sparse = |model: &mut KMeans<f64>| {
        let x = CsrView::new(&indptr, &indices, &data, rows, cols).expect("valid CSR");
        UnsupervisedEstimator::fit(model, x).expect("sparse fit");
    };
    let fit_dense = |model: &mut KMeans<f64>| {
        let x = DenseView::new(&dense, rows, cols).expect("valid dense");
        UnsupervisedEstimator::fit(model, x).expect("dense fit");
    };

    let mut sparse_model = KMeans::<f64>::new(k)
        .with_max_iter(iters)
        .with_tol(0.0)
        .with_seed(1);
    let t = Instant::now();
    fit_sparse(&mut sparse_model);
    let sparse_time = t.elapsed();

    let mut dense_model = KMeans::<f64>::new(k)
        .with_max_iter(iters)
        .with_tol(0.0)
        .with_seed(1);
    let t = Instant::now();
    fit_dense(&mut dense_model);
    let dense_time = t.elapsed();

    let nnz = data.len();
    let csr_bytes = (indptr.len() * 8) + nnz * (8 + 8);
    let dense_bytes = rows * cols * 8;
    println!("rows {rows}, cols {cols}, density {density}, k {k}, iterations {iters}");
    println!(
        "non-zeros {nnz} ({:.2}% of entries)",
        100.0 * nnz as f64 / (rows * cols) as f64
    );
    println!("storage: CSR {csr_bytes} bytes, dense {dense_bytes} bytes");
    println!("sparse fit {:.3} s", sparse_time.as_secs_f64());
    println!("dense fit  {:.3} s", dense_time.as_secs_f64());
    println!(
        "same labels: {}",
        sparse_model.labels() == dense_model.labels()
    );
}
