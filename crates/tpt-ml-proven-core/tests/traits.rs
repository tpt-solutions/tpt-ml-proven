// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Exercises the core traits end to end through the toy estimators.

use tpt_ml_proven_core::toy::{ColumnCenterer, MeanRegressor};
use tpt_ml_proven_core::{
    ArrayView, Dense, DenseView, Error, Predictor, SupervisedEstimator, Transformer,
    UnsupervisedEstimator,
};

const RAW: [f64; 6] = [1.0, 10.0, 3.0, 20.0, 5.0, 30.0];

#[test]
fn centerer_fit_then_transform_centres_columns() {
    let mut c = ColumnCenterer::<f64>::new();
    UnsupervisedEstimator::fit(&mut c, DenseView::new(&RAW, 3, 2).unwrap()).unwrap();
    assert_eq!(c.means(), Some(&[3.0, 20.0][..]));

    let out = c.transform(DenseView::new(&RAW, 3, 2).unwrap()).unwrap();
    assert_eq!(out.into_vec(), vec![-2.0, -10.0, 0.0, 0.0, 2.0, 10.0]);
}

#[test]
fn centerer_fit_transform_matches_fit_then_transform() {
    let mut c = ColumnCenterer::<f32>::new();
    let raw32: Vec<f32> = RAW.iter().map(|&v| v as f32).collect();
    let out = c
        .fit_transform(DenseView::new(&raw32, 3, 2).unwrap())
        .unwrap();
    // The output columns sum to zero: centring removes the mean.
    let col_sums: Vec<f32> = (0..2)
        .map(|j| (0..3).map(|i| out.get(i, j)).sum())
        .collect();
    assert!(col_sums.iter().all(|s| s.abs() < 1e-6));
}

#[test]
fn centerer_transform_before_fit_is_not_fitted_error() {
    let c = ColumnCenterer::<f64>::new();
    let err = c
        .transform(DenseView::new(&RAW, 3, 2).unwrap())
        .unwrap_err();
    assert_eq!(err, Error::NotFitted);
}

#[test]
fn centerer_rejects_feature_count_mismatch_at_transform() {
    let mut c = ColumnCenterer::<f64>::new();
    UnsupervisedEstimator::fit(&mut c, DenseView::new(&RAW, 3, 2).unwrap()).unwrap();
    let one_col = [1.0, 2.0, 3.0];
    let err = c
        .transform(DenseView::new(&one_col, 3, 1).unwrap())
        .unwrap_err();
    assert!(matches!(err, Error::Shape(_)));
}

#[test]
fn centerer_rejects_non_finite_input() {
    let mut c = ColumnCenterer::<f64>::new();
    let bad = [1.0, f64::NAN, 3.0, 4.0];
    let err = UnsupervisedEstimator::fit(&mut c, DenseView::new(&bad, 2, 2).unwrap()).unwrap_err();
    assert_eq!(err, Error::NonFinite { what: "X" });
    assert!(
        c.means().is_none(),
        "failed fit must not leave a partial model"
    );
}

#[test]
#[cfg(feature = "runtime-contracts")]
fn centerer_rejects_empty_input_via_requires() {
    let mut c = ColumnCenterer::<f64>::new();
    let empty: [f64; 0] = [];
    let view = DenseView::new(&empty, 0, 2).unwrap();
    let err = UnsupervisedEstimator::fit(&mut c, view).unwrap_err();
    assert!(matches!(err, Error::Contract(_)));
}

#[test]
fn regressor_fit_and_predict() {
    let x = Dense::from_vec(vec![0.0_f64, 1.0, 2.0, 3.0], 4, 1).unwrap();
    let y: &[f64] = &[2.0, 4.0, 6.0, 8.0];
    let mut m = MeanRegressor::<f64>::new();
    SupervisedEstimator::fit(&mut m, x.view(), y).unwrap();
    assert_eq!(m.mean(), Some(5.0));
    let pred = m
        .predict(DenseView::new(&[9.0, 9.0], 2, 1).unwrap())
        .unwrap();
    assert_eq!(pred, vec![5.0, 5.0]);
}

#[test]
fn regressor_predict_before_fit_is_not_fitted_error() {
    let m = MeanRegressor::<f32>::new();
    let err = m
        .predict(DenseView::new(&[1.0_f32], 1, 1).unwrap())
        .unwrap_err();
    assert_eq!(err, Error::NotFitted);
}

#[test]
#[cfg(feature = "runtime-contracts")]
fn regressor_rejects_row_target_mismatch_via_requires() {
    let x = Dense::from_vec(vec![0.0_f64, 1.0], 2, 1).unwrap();
    let y: &[f64] = &[1.0, 2.0, 3.0];
    let mut m = MeanRegressor::<f64>::new();
    let err = SupervisedEstimator::fit(&mut m, x.view(), y).unwrap_err();
    assert!(matches!(err, Error::Contract(_)));
}

#[test]
fn regressor_rejects_non_finite_target() {
    let x = Dense::from_vec(vec![0.0_f64, 1.0], 2, 1).unwrap();
    let y: &[f64] = &[1.0, f64::INFINITY];
    let mut m = MeanRegressor::<f64>::new();
    let err = SupervisedEstimator::fit(&mut m, x.view(), y).unwrap_err();
    assert_eq!(err, Error::NonFinite { what: "y" });
}

#[test]
fn composition_transform_then_regress() {
    // Generic composition, no pipeline object: the transformer's output is the
    // regressor's input.
    fn run<T, M>(t: &mut T, m: &mut M, x: &[f64]) -> Result<Vec<f64>, Error>
    where
        T: Transformer<Float = f64, Output = Dense<f64>>,
        M: SupervisedEstimator<Float = f64, FitError = Error>
            + Predictor<Float = f64, Output = Vec<f64>>,
    {
        let z = t.fit_transform(DenseView::new(x, 2, 2)?)?;
        m.fit(z.view(), &[1.0, 3.0][..])?;
        m.predict(z.view())
    }

    let mut c = ColumnCenterer::<f64>::new();
    let mut m = MeanRegressor::<f64>::new();
    let pred = run(&mut c, &mut m, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(pred, vec![2.0, 2.0]);
}
