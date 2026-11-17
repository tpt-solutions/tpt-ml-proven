// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! KMeans behaviour: edge cases, determinism, sparse input, and descent.

use tpt_ml_proven_clustering::{Init, KMeans};
use tpt_ml_proven_core::{
    ArrayView, CsrView, Dense, DenseView, Error, Predictor, Transformer, UnsupervisedEstimator,
};

/// Tiny deterministic generator for test data (independent of the crate's RNG).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn random_matrix(seed: u64, rows: usize, cols: usize) -> Vec<f64> {
    let mut g = Lcg(seed);
    (0..rows * cols).map(|_| g.next() * 10.0 - 5.0).collect()
}

fn blobs() -> Vec<f64> {
    // Two well-separated groups of three points each.
    vec![
        0.0, 0.0, 0.1, 0.0, 0.0, 0.1, // near (0, 0)
        10.0, 10.0, 10.1, 10.0, 10.0, 10.1, // near (10, 10)
    ]
}

fn fit_f64(data: &[f64], rows: usize, cols: usize, k: usize, seed: u64) -> KMeans<f64> {
    let x = DenseView::new(data, rows, cols).unwrap();
    let mut km = KMeans::<f64>::new(k).with_seed(seed);
    UnsupervisedEstimator::fit(&mut km, x).unwrap();
    km
}

/// Relabels a clustering in order of first appearance, so two clusterings can be
/// compared up to a permutation of the labels.
fn canonical(labels: &[usize]) -> Vec<usize> {
    let mut seen: Vec<usize> = Vec::new();
    labels
        .iter()
        .map(|&l| match seen.iter().position(|&s| s == l) {
            Some(p) => p,
            None => {
                seen.push(l);
                seen.len() - 1
            }
        })
        .collect()
}

#[test]
fn recovers_two_well_separated_blobs() {
    let km = fit_f64(&blobs(), 6, 2, 2, 1);
    assert_eq!(canonical(km.labels().unwrap()), vec![0, 0, 0, 1, 1, 1]);
    assert_eq!(km.converged(), Some(true));
    assert!(km.inertia().unwrap() < 0.1);
}

#[test]
fn k_equals_one_centre_is_the_mean_and_inertia_is_total_variance() {
    let data = [1.0_f64, 2.0, 3.0, 6.0];
    let km = fit_f64(&data, 4, 1, 1, 0);
    let c = km.cluster_centers().unwrap();
    assert!((c.get(0, 0) - 3.0).abs() < 1e-12);
    // Sum of squared deviations from 3: 4 + 1 + 0 + 9.
    assert!((km.inertia().unwrap() - 14.0).abs() < 1e-9);
}

#[test]
fn k_equals_n_gives_zero_inertia_on_distinct_points() {
    let data = [0.0_f64, 0.0, 1.0, 5.0, 3.0, -2.0];
    let km = fit_f64(&data, 3, 2, 3, 4);
    assert!(km.inertia().unwrap().abs() < 1e-12);
    let mut labels = canonical(km.labels().unwrap());
    labels.sort_unstable();
    assert_eq!(labels, vec![0, 1, 2]);
}

#[test]
fn duplicate_points_fit_without_nan_and_with_zero_inertia() {
    let data = [4.0_f64, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0];
    for init in [Init::KMeansPlusPlus, Init::Random] {
        let x = DenseView::new(&data, 4, 2).unwrap();
        let mut km = KMeans::<f64>::new(3).with_init(init).with_seed(3);
        UnsupervisedEstimator::fit(&mut km, x).unwrap();
        assert_eq!(km.inertia(), Some(0.0));
        for &v in km.cluster_centers().unwrap().clone().into_vec().iter() {
            assert!(v.is_finite());
        }
    }
}

#[test]
fn nan_and_infinity_are_rejected() {
    for bad in [f64::NAN, f64::INFINITY] {
        let data = [0.0_f64, bad, 1.0, 2.0];
        let mut km = KMeans::<f64>::new(1);
        let err =
            UnsupervisedEstimator::fit(&mut km, DenseView::new(&data, 2, 2).unwrap()).unwrap_err();
        assert_eq!(err, Error::NonFinite { what: "X" });
        assert!(
            km.labels().is_none(),
            "a failed fit leaves the estimator unfitted"
        );
    }
}

#[test]
fn invalid_parameters_are_rejected_with_a_name() {
    let data = [0.0_f64, 1.0, 2.0];
    let cases: Vec<(KMeans<f64>, &str)> = vec![
        (KMeans::new(0), "n_clusters"),
        (KMeans::new(4), "n_clusters"),
        (KMeans::new(1).with_max_iter(0), "max_iter"),
        (KMeans::new(1).with_tol(-1.0), "tol"),
        (KMeans::new(1).with_tol(f64::NAN), "tol"),
        (KMeans::new(1).with_threads(0), "threads"),
    ];
    for (mut km, name) in cases {
        let err =
            UnsupervisedEstimator::fit(&mut km, DenseView::new(&data, 3, 1).unwrap()).unwrap_err();
        assert!(
            matches!(err, Error::InvalidParameter { name: n, .. } if n == name),
            "{name}: got {err:?}"
        );
    }
}

#[test]
fn empty_input_is_rejected() {
    let empty: [f64; 0] = [];
    let mut km = KMeans::<f64>::new(1);
    let err =
        UnsupervisedEstimator::fit(&mut km, DenseView::new(&empty, 0, 2).unwrap()).unwrap_err();
    assert_eq!(err, Error::EmptyInput);
}

#[test]
fn same_seed_gives_same_model_for_both_initialisations() {
    let data = random_matrix(11, 60, 3);
    for init in [Init::KMeansPlusPlus, Init::Random] {
        let run = || {
            let mut km = KMeans::<f64>::new(4).with_init(init).with_seed(99);
            UnsupervisedEstimator::fit(&mut km, DenseView::new(&data, 60, 3).unwrap()).unwrap();
            km
        };
        let (a, b) = (run(), run());
        assert_eq!(a.labels(), b.labels());
        assert_eq!(a.inertia(), b.inertia());
        assert_eq!(a.cluster_centers(), b.cluster_centers());
    }
}

#[test]
fn predict_matches_training_labels_and_transform_has_the_right_shape() {
    let data = blobs();
    let km = fit_f64(&data, 6, 2, 2, 1);
    let x = DenseView::new(&data, 6, 2).unwrap();
    assert_eq!(km.predict(x).unwrap(), km.labels().unwrap().to_vec());

    let d = km.transform(DenseView::new(&data, 6, 2).unwrap()).unwrap();
    assert_eq!((d.rows(), d.cols()), (6, 2));
    // Each point is closest to the centre it is labelled with.
    for (r, &label) in km.labels().unwrap().iter().enumerate() {
        let other = 1 - label;
        assert!(d.get(r, label) <= d.get(r, other));
        assert!(d.get(r, label) >= 0.0);
    }
}

#[test]
fn predict_and_transform_before_fit_are_not_fitted() {
    let km = KMeans::<f64>::new(2);
    let x = DenseView::new(&[0.0_f64, 1.0], 2, 1).unwrap();
    assert_eq!(km.predict(x).unwrap_err(), Error::NotFitted);
    assert_eq!(km.transform(x).unwrap_err(), Error::NotFitted);
}

#[test]
fn predict_rejects_feature_count_mismatch() {
    let km = fit_f64(&blobs(), 6, 2, 2, 1);
    let one_col = [0.0_f64, 1.0, 2.0];
    let err = km
        .predict(DenseView::new(&one_col, 3, 1).unwrap())
        .unwrap_err();
    assert!(matches!(err, Error::Shape(_)));
}

#[test]
fn sparse_input_gives_the_same_model_as_dense() {
    // A sparse matrix with explicit zeros removed from the stored structure.
    let dense = [
        1.0_f64, 0.0, 0.0, //
        1.5, 0.0, 0.0, //
        0.0, 0.0, 9.0, //
        0.0, 0.0, 9.5,
    ];
    let indptr = [0, 1, 2, 3, 4];
    let indices = [0, 0, 2, 2];
    let data = [1.0_f64, 1.5, 9.0, 9.5];
    let sparse = CsrView::new(&indptr, &indices, &data, 4, 3).unwrap();

    let mut ks = KMeans::<f64>::new(2).with_seed(5);
    UnsupervisedEstimator::fit(&mut ks, sparse).unwrap();
    let kd = fit_f64(&dense, 4, 3, 2, 5);

    assert_eq!(
        canonical(ks.labels().unwrap()),
        canonical(kd.labels().unwrap())
    );
    assert!((ks.inertia().unwrap() - kd.inertia().unwrap()).abs() < 1e-12);
}

#[test]
fn works_for_f32_and_agrees_with_f64_on_easy_data() {
    let data64 = blobs();
    let data32: Vec<f32> = data64.iter().map(|&v| v as f32).collect();
    let mut k32 = KMeans::<f32>::new(2).with_seed(1);
    UnsupervisedEstimator::fit(&mut k32, DenseView::new(&data32, 6, 2).unwrap()).unwrap();
    let k64 = fit_f64(&data64, 6, 2, 2, 1);
    assert_eq!(
        canonical(k32.labels().unwrap()),
        canonical(k64.labels().unwrap())
    );
}

#[test]
fn inertia_never_increases_as_iterations_are_allowed() {
    // For a fixed seed (fixed initial centres), a longer run can only go lower, and
    // the contract's monotone-descent clause says so per iteration.
    for seed in 0..25_u64 {
        let rows = 40;
        let data = random_matrix(seed + 100, rows, 2);
        let mut previous = f64::INFINITY;
        for max_iter in 1..=8 {
            let mut km = KMeans::<f64>::new(3)
                .with_seed(seed)
                .with_max_iter(max_iter)
                .with_tol(0.0);
            UnsupervisedEstimator::fit(&mut km, DenseView::new(&data, rows, 2).unwrap()).unwrap();
            let inertia = km.inertia().unwrap();
            assert!(
                inertia <= previous + 1e-9,
                "seed {seed}, max_iter {max_iter}: {inertia} > {previous}"
            );
            previous = inertia;
        }
    }
}

#[test]
fn random_inputs_with_duplicates_and_extreme_k_stay_finite_and_valid() {
    for seed in 0..40_u64 {
        let rows = 2 + (seed as usize % 9);
        let cols = 1 + (seed as usize % 3);
        // Small integer coordinates force many duplicate points.
        let data: Vec<f64> = random_matrix(seed, rows, cols)
            .iter()
            .map(|v| v.round())
            .collect();
        for k in [1, rows / 2, rows].into_iter().filter(|&k| k >= 1) {
            for init in [Init::KMeansPlusPlus, Init::Random] {
                let mut km = KMeans::<f64>::new(k).with_init(init).with_seed(seed);
                UnsupervisedEstimator::fit(&mut km, DenseView::new(&data, rows, cols).unwrap())
                    .unwrap();
                let labels = km.labels().unwrap();
                assert_eq!(labels.len(), rows);
                assert!(labels.iter().all(|&l| l < k));
                assert!(km.inertia().unwrap().is_finite());
                assert!(km
                    .cluster_centers()
                    .unwrap()
                    .clone()
                    .into_vec()
                    .iter()
                    .all(|v| v.is_finite()));
            }
        }
    }
}

#[test]
fn stops_at_max_iter_and_reports_not_converged() {
    let data = random_matrix(3, 50, 2);
    let mut km = KMeans::<f64>::new(4)
        .with_seed(1)
        .with_max_iter(1)
        .with_tol(0.0);
    UnsupervisedEstimator::fit(&mut km, DenseView::new(&data, 50, 2).unwrap()).unwrap();
    assert_eq!(km.n_iter(), Some(1));
    assert_eq!(km.converged(), Some(false));
}

#[test]
fn transform_via_fit_transform_matches_fit_then_transform() {
    let data = blobs();
    let mut a = KMeans::<f64>::new(2).with_seed(1);
    let da = a
        .fit_transform(DenseView::new(&data, 6, 2).unwrap())
        .unwrap();
    let b = fit_f64(&data, 6, 2, 2, 1);
    let db = b.transform(DenseView::new(&data, 6, 2).unwrap()).unwrap();
    assert_eq!(da, db);
    let _: Dense<f64> = da;
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_assignment_matches_sequential_exactly() {
    let data = random_matrix(21, 500, 4);
    let run = |threads: usize| {
        let mut km = KMeans::<f64>::new(5).with_seed(7).with_threads(threads);
        UnsupervisedEstimator::fit(&mut km, DenseView::new(&data, 500, 4).unwrap()).unwrap();
        km
    };
    let (one, four) = (run(1), run(4));
    assert_eq!(one.labels(), four.labels());
    assert_eq!(one.inertia(), four.inertia());
    assert_eq!(one.cluster_centers(), four.cluster_centers());
}

#[test]
fn explicit_initial_centres_are_used_and_length_checked() {
    let data = blobs();
    let x = DenseView::new(&data, 6, 2).unwrap();
    // Starting exactly at the two blob centres gives the obvious partition.
    let mut km = KMeans::<f64>::new(2).with_initial_centers(vec![0.0, 0.0, 10.0, 10.0]);
    UnsupervisedEstimator::fit(&mut km, x).unwrap();
    assert_eq!(canonical(km.labels().unwrap()), vec![0, 0, 0, 1, 1, 1]);

    let mut bad = KMeans::<f64>::new(2).with_initial_centers(vec![0.0, 0.0, 10.0]);
    let err =
        UnsupervisedEstimator::fit(&mut bad, DenseView::new(&data, 6, 2).unwrap()).unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidParameter {
            name: "initial_centers",
            ..
        }
    ));
}
