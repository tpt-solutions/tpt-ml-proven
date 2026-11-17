// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Times `KMeans::fit` on a raw little-endian `f32` matrix, for comparison with
//! scikit-learn. The data file is written by `tools/parity/bench_kmeans.py`.
//!
//! Run with optimisations, otherwise the numbers mean nothing:
//!
//! ```text
//! cargo run --release -p tpt-ml-proven-clustering --example bench_kmeans \
//!     --features parallel -- <data.bin> <rows> <cols> <k> <iters>
//! ```

use std::time::Instant;

use tpt_ml_proven_clustering::KMeans;
use tpt_ml_proven_core::{DenseView, UnsupervisedEstimator};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        eprintln!("usage: bench_kmeans <data.bin> <rows> <cols> <k> <iters>");
        std::process::exit(2);
    }
    let path = &args[1];
    let rows: usize = args[2].parse().expect("rows");
    let cols: usize = args[3].parse().expect("cols");
    let k: usize = args[4].parse().expect("k");
    let iters: usize = args[5].parse().expect("iters");

    let bytes = std::fs::read(path).expect("read data file");
    let (chunks, _) = bytes.as_chunks::<4>();
    let data: Vec<f32> = chunks.iter().map(|c| f32::from_le_bytes(*c)).collect();
    assert_eq!(
        data.len(),
        rows * cols,
        "data file does not match rows x cols"
    );

    // The same starting centres as the scikit-learn run: the first k rows.
    let init: Vec<f32> = data[..k * cols].to_vec();

    let threads: Vec<usize> = if cfg!(feature = "parallel") {
        vec![
            1,
            std::thread::available_parallelism().map_or(1, |n| n.get()),
        ]
    } else {
        vec![1]
    };
    for t in threads {
        let mut best = f64::INFINITY;
        let mut n_iter = None;
        for _ in 0..3 {
            let mut km = KMeans::<f32>::new(k)
                .with_initial_centers(init.clone())
                .with_tol(0.0)
                .with_max_iter(iters)
                .with_threads(t);
            let start = Instant::now();
            UnsupervisedEstimator::fit(&mut km, DenseView::new(&data, rows, cols).unwrap())
                .expect("fit");
            best = best.min(start.elapsed().as_secs_f64());
            n_iter = km.n_iter();
        }
        println!("{{\"impl\": \"tpt-ml-proven\", \"threads\": {t}, \"seconds\": {best:.4}, \"n_iter\": {}}}", n_iter.unwrap_or(0));
    }
}
