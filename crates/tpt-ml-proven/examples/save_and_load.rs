// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Fit KMeans on CSV text, save the model to a file, load it back, and check that it labels
//! the data the same way.
//!
//! Run with `cargo run -p tpt-ml-proven --example save_and_load`.

use tpt_ml_proven::prelude::*;

const CSV: &str = "x,y\n0.0,0.0\n0.1,0.0\n0.0,0.1\n10.0,10.0\n10.1,10.0\n10.0,10.1\n";

fn parse(text: &str) -> (Vec<f64>, usize, usize) {
    let mut data = Vec::new();
    let mut rows = 0;
    let mut cols = 0;
    for line in text.lines().skip(1) {
        let values: Vec<f64> = line.split(',').map(|v| v.trim().parse().unwrap()).collect();
        cols = values.len();
        data.extend(values);
        rows += 1;
    }
    (data, rows, cols)
}

fn main() {
    let (data, rows, cols) = parse(CSV);
    let x = DenseView::new(&data, rows, cols).expect("a rectangular matrix");

    let mut model = KMeans::<f64>::new(2).with_seed(1);
    UnsupervisedEstimator::fit(&mut model, x).expect("KMeans fits");
    let labels = Predictor::predict(&model, DenseView::new(&data, rows, cols).unwrap()).unwrap();
    println!("fitted labels: {labels:?}");

    // Save to a file, as a user would, then load it in a fresh value.
    let path = std::env::temp_dir().join("tpt-ml-proven-example.tpml");
    std::fs::write(&path, model.to_bytes().expect("saving")).expect("writing the file");
    let bytes = std::fs::read(&path).expect("reading the file");
    let loaded = KMeans::<f64>::from_bytes(&bytes).expect("loading the saved model");
    let again = Predictor::predict(&loaded, DenseView::new(&data, rows, cols).unwrap()).unwrap();
    println!("loaded labels: {again:?}");
    assert_eq!(
        labels, again,
        "a loaded model labels the data as the original did"
    );
    let _ = std::fs::remove_file(&path);
    println!(
        "saved to {} and loaded back with identical labels",
        path.display()
    );
}
