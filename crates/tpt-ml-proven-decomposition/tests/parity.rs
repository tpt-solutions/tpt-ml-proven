// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Parity with scikit-learn's `PCA(svd_solver="full")`. Reference values come from
//! committed fixtures in `tools/parity/fixtures`.
//!
//! Normalisation: the sign of each principal axis is not fixed by the method, and
//! scikit-learn and this crate choose differently (sklearn flips by the sign of the
//! left singular vectors, this crate by the largest entry). Each reference component
//! is therefore flipped to agree in sign with ours before comparing. Nothing else is
//! normalised.

use std::collections::HashMap;
use std::path::PathBuf;

use tpt_ml_proven_core::{ArrayView, DenseView, UnsupervisedEstimator};
use tpt_ml_proven_decomposition::Pca;

const TOL: f64 = 1e-9;

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

fn floats(fx: &HashMap<String, Vec<String>>, key: &str) -> Vec<f64> {
    fx[key].iter().map(|s| s.parse().unwrap()).collect()
}

fn assert_close(actual: f64, expected: f64, what: &str) {
    let scale = 1.0 + expected.abs();
    assert!(
        (actual - expected).abs() <= TOL * scale,
        "{what}: {actual} vs scikit-learn {expected}"
    );
}

fn check_pca(name: &str) {
    let fx = load(name);
    let rows: usize = fx["rows"][0].parse().unwrap();
    let cols: usize = fx["cols"][0].parse().unwrap();
    let k: usize = fx["n_components"][0].parse().unwrap();
    let x = floats(&fx, "x");

    let mut pca = Pca::<f64>::new(k);
    UnsupervisedEstimator::fit(&mut pca, DenseView::new(&x, rows, cols).unwrap()).unwrap();

    for (j, &m) in floats(&fx, "mean").iter().enumerate() {
        assert_close(pca.mean().unwrap()[j], m, &format!("{name}: mean[{j}]"));
    }
    let ev = floats(&fx, "explained_variance");
    for (i, &e) in ev.iter().enumerate() {
        assert_close(
            pca.explained_variance().unwrap()[i],
            e,
            &format!("{name}: ev[{i}]"),
        );
    }
    let ratio = floats(&fx, "explained_variance_ratio");
    for (i, &r) in ratio.iter().enumerate() {
        assert_close(
            pca.explained_variance_ratio().unwrap()[i],
            r,
            &format!("{name}: ratio[{i}]"),
        );
    }

    let ours = pca.components().unwrap();
    let theirs = floats(&fx, "components");
    for c in 0..k {
        let reference = &theirs[c * cols..(c + 1) * cols];
        let dot: f64 = (0..cols).map(|j| ours.get(c, j) * reference[j]).sum();
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        for (j, &r) in reference.iter().enumerate() {
            assert_close(
                ours.get(c, j),
                sign * r,
                &format!("{name}: component[{c}][{j}]"),
            );
        }
    }
}

#[test]
fn pca_full_rank_matches_scikit_learn() {
    check_pca("pca_correlated_full");
}

#[test]
fn pca_truncated_matches_scikit_learn() {
    check_pca("pca_correlated_k3");
}

/// TruncatedSVD (ARPACK in scikit-learn): singular values exactly, components up to sign.
#[test]
fn truncated_svd_matches_scikit_learn_arpack() {
    use tpt_ml_proven_decomposition::TruncatedSvd;
    let fx = load("svd_arpack_k2");
    let rows: usize = fx["rows"][0].parse().unwrap();
    let cols: usize = fx["cols"][0].parse().unwrap();
    let k: usize = fx["n_components"][0].parse().unwrap();
    let x = floats(&fx, "x");
    let mut svd = TruncatedSvd::<f64>::new(k);
    UnsupervisedEstimator::fit(&mut svd, DenseView::new(&x, rows, cols).unwrap()).unwrap();
    for (i, &s) in floats(&fx, "singular_values").iter().enumerate() {
        assert_close(svd.singular_values().unwrap()[i], s, &format!("sv[{i}]"));
    }
    let ours = svd.components().unwrap();
    let theirs = floats(&fx, "components");
    for c in 0..k {
        let reference = &theirs[c * cols..(c + 1) * cols];
        let dot: f64 = (0..cols).map(|j| ours.get(c, j) * reference[j]).sum();
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        for (j, &r) in reference.iter().enumerate() {
            assert_close(ours.get(c, j), sign * r, &format!("component[{c}][{j}]"));
        }
    }
}
