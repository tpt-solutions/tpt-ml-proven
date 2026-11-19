// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Estimate how well a ridge regression generalises, with 5-fold cross-validation and R².
//!
//! Run with `cargo run -p tpt-ml-proven --example cross_validate`.

use tpt_ml_proven::metrics;
use tpt_ml_proven::model_selection::{cross_val_score, KFold};
use tpt_ml_proven::prelude::*;

fn main() {
    // y = 2·x0 − x1 + 3 plus a small deterministic wobble, for 50 samples.
    let rows = 50;
    let mut x = Vec::with_capacity(rows * 2);
    let mut y = Vec::with_capacity(rows);
    for i in 0..rows {
        let x0 = (i as f64 * 0.37).sin() * 4.0;
        let x1 = (i as f64 * 0.11).cos() * 3.0;
        let wobble = ((i * 7) % 5) as f64 * 0.01;
        x.extend([x0, x1]);
        y.push(2.0 * x0 - x1 + 3.0 + wobble);
    }
    let view = DenseView::new(&x, rows, 2).expect("a rectangular matrix");

    let scores = cross_val_score(
        &view,
        &y,
        &KFold::new(5).with_shuffle(42),
        || Ridge::<f64>::new(0.1),
        |model, test_x, test_y| {
            let pred = Predictor::predict(model, test_x.view())?;
            metrics::r2(test_y, &pred)
        },
    )
    .expect("cross-validation runs");

    for (fold, score) in scores.iter().enumerate() {
        println!("fold {}: R² = {score:.4}", fold + 1);
    }
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    println!("mean R² over {} folds: {mean:.4}", scores.len());
    assert!(mean > 0.99, "a linear signal should cross-validate well");
}
