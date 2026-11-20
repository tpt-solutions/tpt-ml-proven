// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! C ABI for the trees and forests: single trees, gradient boosting and random forests, for
//! regression and classification.
//!
//! A `max_features` of `0` means the default for the model. A `criterion` is `TpmlCriterion`.

use std::ffi::c_double;

use tpt_ml_proven_core::metrics::accuracy;
use tpt_ml_proven_core::{DenseView, Predictor, SupervisedEstimator};
use tpt_ml_proven_trees::{
    Criterion, DecisionTreeClassifier, DecisionTreeRegressor, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
};

use super::models::{null_handle, targets};
use super::{input, output, run, Failure, TpmlStatus};

/// Impurity measure for the classification trees and forests. C callers construct these.
#[repr(C)]
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpmlCriterion {
    /// Gini impurity.
    Gini = 0,
    /// Shannon entropy.
    Entropy = 1,
}

impl From<TpmlCriterion> for Criterion {
    fn from(c: TpmlCriterion) -> Self {
        match c {
            TpmlCriterion::Gini => Criterion::Gini,
            TpmlCriterion::Entropy => Criterion::Entropy,
        }
    }
}

fn optional_features(max_features: usize) -> Option<usize> {
    (max_features > 0).then_some(max_features)
}

/// A leaf limit of zero means no limit.
fn optional_leaves(max_leaf_nodes: usize) -> Option<usize> {
    (max_leaf_nodes > 0).then_some(max_leaf_nodes)
}

// ---------------------------------------------------------------- decision tree regressor

/// Opaque decision tree regressor handle.
pub struct TpmlDecisionTreeRegressor {
    max_depth: usize,
    min_samples_leaf: usize,
    min_samples_split: usize,
    max_leaf_nodes: usize,
    fitted: Option<DecisionTreeRegressor>,
}

/// Creates a regression tree. Returns null if `max_depth` or `min_samples_leaf` is zero.
#[no_mangle]
pub extern "C" fn tpml_decision_tree_regressor_new(
    max_depth: usize,
    min_samples_leaf: usize,
) -> *mut TpmlDecisionTreeRegressor {
    if max_depth == 0 || min_samples_leaf == 0 {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(TpmlDecisionTreeRegressor {
        max_depth,
        min_samples_leaf,
        min_samples_split: 2,
        max_leaf_nodes: 0,
        fitted: None,
    }))
}

/// Frees a regression tree handle. Null is ignored.
///
/// # Safety
/// `h` must be null or a live handle from `tpml_decision_tree_regressor_new`.
#[no_mangle]
pub unsafe extern "C" fn tpml_decision_tree_regressor_free(h: *mut TpmlDecisionTreeRegressor) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Fits to a row-major `rows × cols` matrix and `rows` targets.
///
/// # Safety
/// `data` is valid for `rows * cols` reads and `y` for `rows` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_decision_tree_regressor_fit(
    h: *mut TpmlDecisionTreeRegressor,
    data: *const c_double,
    rows: usize,
    cols: usize,
    y: *const c_double,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        let x = input(data, rows, cols)?;
        let t = targets(y, rows)?;
        let mut m = DecisionTreeRegressor::new(h.max_depth)
            .with_min_samples_leaf(h.min_samples_leaf)
            .with_min_samples_split(h.min_samples_split)
            .with_max_leaf_nodes(optional_leaves(h.max_leaf_nodes));
        SupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols)?, t)?;
        h.fitted = Some(m);
        Ok(())
    })
}

/// Predicts for a row-major `rows × cols` matrix. `out` receives `rows` values.
///
/// # Safety
/// `data` is valid for `rows * cols` reads and `out` for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_decision_tree_regressor_predict(
    h: *const TpmlDecisionTreeRegressor,
    data: *const c_double,
    rows: usize,
    cols: usize,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h
            .fitted
            .as_ref()
            .ok_or(tpt_ml_proven_core::error::Error::NotFitted)?;
        let x = input(data, rows, cols)?;
        let p = Predictor::predict(m, DenseView::new(x, rows, cols)?)?;
        output(out, out_len, &p)
    })
}

// ---------------------------------------------------------------- decision tree classifier

/// Opaque decision tree classifier handle.
pub struct TpmlDecisionTreeClassifier {
    max_depth: usize,
    min_samples_leaf: usize,
    min_samples_split: usize,
    max_leaf_nodes: usize,
    criterion: Criterion,
    fitted: Option<DecisionTreeClassifier>,
}

/// Creates a classification tree. Returns null if `max_depth` or `min_samples_leaf` is zero.
#[no_mangle]
pub extern "C" fn tpml_decision_tree_classifier_new(
    max_depth: usize,
    min_samples_leaf: usize,
    criterion: TpmlCriterion,
) -> *mut TpmlDecisionTreeClassifier {
    if max_depth == 0 || min_samples_leaf == 0 {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(TpmlDecisionTreeClassifier {
        max_depth,
        min_samples_leaf,
        min_samples_split: 2,
        max_leaf_nodes: 0,
        criterion: criterion.into(),
        fitted: None,
    }))
}

/// Frees a classification tree handle. Null is ignored.
///
/// # Safety
/// `h` must be null or a live handle from `tpml_decision_tree_classifier_new`.
#[no_mangle]
pub unsafe extern "C" fn tpml_decision_tree_classifier_free(h: *mut TpmlDecisionTreeClassifier) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Fits to a row-major `rows × cols` matrix and `rows` class labels.
///
/// # Safety
/// `data` is valid for `rows * cols` reads and `y` for `rows` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_decision_tree_classifier_fit(
    h: *mut TpmlDecisionTreeClassifier,
    data: *const c_double,
    rows: usize,
    cols: usize,
    y: *const c_double,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        let x = input(data, rows, cols)?;
        let t = targets(y, rows)?;
        let mut m = DecisionTreeClassifier::new(h.max_depth)
            .with_min_samples_leaf(h.min_samples_leaf)
            .with_min_samples_split(h.min_samples_split)
            .with_max_leaf_nodes(optional_leaves(h.max_leaf_nodes))
            .with_criterion(h.criterion);
        SupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols)?, t)?;
        h.fitted = Some(m);
        Ok(())
    })
}

/// Predicts class labels for a row-major `rows × cols` matrix. `out` receives `rows` values.
///
/// # Safety
/// `data` is valid for `rows * cols` reads and `out` for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_decision_tree_classifier_predict(
    h: *const TpmlDecisionTreeClassifier,
    data: *const c_double,
    rows: usize,
    cols: usize,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h
            .fitted
            .as_ref()
            .ok_or(tpt_ml_proven_core::error::Error::NotFitted)?;
        let x = input(data, rows, cols)?;
        let p = Predictor::predict(m, DenseView::new(x, rows, cols)?)?;
        output(out, out_len, &p)
    })
}

/// Writes the number of training classes into `out`.
///
/// # Safety
/// `h` is a live handle and `out` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn tpml_decision_tree_classifier_n_classes(
    h: *const TpmlDecisionTreeClassifier,
    out: *mut usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h
            .fitted
            .as_ref()
            .ok_or(tpt_ml_proven_core::error::Error::NotFitted)?;
        output(out, 1, &[m.classes().map_or(0, <[f64]>::len)])
    })
}

/// Writes the class labels, sorted, into `out`. `out_len` must equal the class count.
///
/// # Safety
/// `h` is a live handle and `out` valid for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_decision_tree_classifier_classes(
    h: *const TpmlDecisionTreeClassifier,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h
            .fitted
            .as_ref()
            .ok_or(tpt_ml_proven_core::error::Error::NotFitted)?;
        output(out, out_len, m.classes().unwrap_or(&[]))
    })
}

// ---------------------------------------------------------------- gradient boosting

/// Opaque gradient boosting handle.
pub struct TpmlGradientBoosting {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_leaf: usize,
    min_samples_split: usize,
    max_leaf_nodes: usize,
    subsample: f64,
    max_features: usize,
    seed: u64,
    fitted: Option<GradientBoostingRegressor>,
}

/// Creates a gradient boosting regressor. Returns null for an invalid setting: zero trees,
/// depth or leaf size, a learning rate outside `(0, 1]`, or a subsample outside `(0, 1]`.
/// `max_features` of zero means all features.
#[no_mangle]
pub extern "C" fn tpml_gradient_boosting_new(
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_leaf: usize,
    subsample: f64,
    max_features: usize,
    seed: u64,
) -> *mut TpmlGradientBoosting {
    let valid = n_estimators > 0
        && max_depth > 0
        && min_samples_leaf > 0
        && learning_rate.is_finite()
        && learning_rate > 0.0
        && learning_rate <= 1.0
        && subsample.is_finite()
        && subsample > 0.0
        && subsample <= 1.0;
    if !valid {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(TpmlGradientBoosting {
        n_estimators,
        learning_rate,
        max_depth,
        min_samples_leaf,
        min_samples_split: 2,
        max_leaf_nodes: 0,
        subsample,
        max_features,
        seed,
        fitted: None,
    }))
}

/// Frees a gradient boosting handle. Null is ignored.
///
/// # Safety
/// `h` must be null or a live handle from `tpml_gradient_boosting_new`.
#[no_mangle]
pub unsafe extern "C" fn tpml_gradient_boosting_free(h: *mut TpmlGradientBoosting) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Fits to a row-major `rows × cols` matrix and `rows` targets.
///
/// # Safety
/// `data` is valid for `rows * cols` reads and `y` for `rows` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_gradient_boosting_fit(
    h: *mut TpmlGradientBoosting,
    data: *const c_double,
    rows: usize,
    cols: usize,
    y: *const c_double,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        let x = input(data, rows, cols)?;
        let t = targets(y, rows)?;
        let mut m = GradientBoostingRegressor::new(h.n_estimators, h.max_depth)
            .with_learning_rate(h.learning_rate)
            .with_min_samples_leaf(h.min_samples_leaf)
            .with_min_samples_split(h.min_samples_split)
            .with_max_leaf_nodes(optional_leaves(h.max_leaf_nodes))
            .with_subsample(h.subsample)
            .with_seed(h.seed);
        if let Some(f) = optional_features(h.max_features) {
            m = m.with_max_features(f);
        }
        SupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols)?, t)?;
        h.fitted = Some(m);
        Ok(())
    })
}

/// Predicts for a row-major `rows × cols` matrix. `out` receives `rows` values.
///
/// # Safety
/// `data` is valid for `rows * cols` reads and `out` for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_gradient_boosting_predict(
    h: *const TpmlGradientBoosting,
    data: *const c_double,
    rows: usize,
    cols: usize,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h
            .fitted
            .as_ref()
            .ok_or(tpt_ml_proven_core::error::Error::NotFitted)?;
        let x = input(data, rows, cols)?;
        let p = Predictor::predict(m, DenseView::new(x, rows, cols)?)?;
        output(out, out_len, &p)
    })
}

// ---------------------------------------------------------------- random forests

/// Opaque random forest regressor handle.
pub struct TpmlRandomForestRegressor {
    n_estimators: usize,
    max_depth: usize,
    min_samples_leaf: usize,
    min_samples_split: usize,
    max_leaf_nodes: usize,
    max_features: usize,
    seed: u64,
    fitted: Option<RandomForestRegressor>,
}

/// Creates a random forest regressor. Returns null if `n_estimators`, `max_depth` or
/// `min_samples_leaf` is zero. `max_features` of zero means all features.
#[no_mangle]
pub extern "C" fn tpml_random_forest_regressor_new(
    n_estimators: usize,
    max_depth: usize,
    min_samples_leaf: usize,
    max_features: usize,
    seed: u64,
) -> *mut TpmlRandomForestRegressor {
    if n_estimators == 0 || max_depth == 0 || min_samples_leaf == 0 {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(TpmlRandomForestRegressor {
        n_estimators,
        max_depth,
        min_samples_leaf,
        min_samples_split: 2,
        max_leaf_nodes: 0,
        max_features,
        seed,
        fitted: None,
    }))
}

/// Frees a random forest regressor handle. Null is ignored.
///
/// # Safety
/// `h` must be null or a live handle from `tpml_random_forest_regressor_new`.
#[no_mangle]
pub unsafe extern "C" fn tpml_random_forest_regressor_free(h: *mut TpmlRandomForestRegressor) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Fits to a row-major `rows × cols` matrix and `rows` targets.
///
/// # Safety
/// `data` is valid for `rows * cols` reads and `y` for `rows` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_random_forest_regressor_fit(
    h: *mut TpmlRandomForestRegressor,
    data: *const c_double,
    rows: usize,
    cols: usize,
    y: *const c_double,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        let x = input(data, rows, cols)?;
        let t = targets(y, rows)?;
        let mut m = RandomForestRegressor::new(h.n_estimators, h.max_depth)
            .with_min_samples_leaf(h.min_samples_leaf)
            .with_min_samples_split(h.min_samples_split)
            .with_max_leaf_nodes(optional_leaves(h.max_leaf_nodes))
            .with_seed(h.seed);
        if let Some(f) = optional_features(h.max_features) {
            m = m.with_max_features(f);
        }
        SupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols)?, t)?;
        h.fitted = Some(m);
        Ok(())
    })
}

/// Predicts for a row-major `rows × cols` matrix. `out` receives `rows` values.
///
/// # Safety
/// `data` is valid for `rows * cols` reads and `out` for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_random_forest_regressor_predict(
    h: *const TpmlRandomForestRegressor,
    data: *const c_double,
    rows: usize,
    cols: usize,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h
            .fitted
            .as_ref()
            .ok_or(tpt_ml_proven_core::error::Error::NotFitted)?;
        let x = input(data, rows, cols)?;
        let p = Predictor::predict(m, DenseView::new(x, rows, cols)?)?;
        output(out, out_len, &p)
    })
}

/// Opaque random forest classifier handle.
pub struct TpmlRandomForestClassifier {
    n_estimators: usize,
    max_depth: usize,
    min_samples_leaf: usize,
    min_samples_split: usize,
    max_leaf_nodes: usize,
    criterion: Criterion,
    max_features: usize,
    seed: u64,
    fitted: Option<RandomForestClassifier>,
}

/// Creates a random forest classifier. Returns null if `n_estimators`, `max_depth` or
/// `min_samples_leaf` is zero. `max_features` of zero means the square root of the feature count.
#[no_mangle]
pub extern "C" fn tpml_random_forest_classifier_new(
    n_estimators: usize,
    max_depth: usize,
    min_samples_leaf: usize,
    criterion: TpmlCriterion,
    max_features: usize,
    seed: u64,
) -> *mut TpmlRandomForestClassifier {
    if n_estimators == 0 || max_depth == 0 || min_samples_leaf == 0 {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(TpmlRandomForestClassifier {
        n_estimators,
        max_depth,
        min_samples_leaf,
        min_samples_split: 2,
        max_leaf_nodes: 0,
        criterion: criterion.into(),
        max_features,
        seed,
        fitted: None,
    }))
}

/// Frees a random forest classifier handle. Null is ignored.
///
/// # Safety
/// `h` must be null or a live handle from `tpml_random_forest_classifier_new`.
#[no_mangle]
pub unsafe extern "C" fn tpml_random_forest_classifier_free(h: *mut TpmlRandomForestClassifier) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Fits to a row-major `rows × cols` matrix and `rows` class labels.
///
/// # Safety
/// `data` is valid for `rows * cols` reads and `y` for `rows` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_random_forest_classifier_fit(
    h: *mut TpmlRandomForestClassifier,
    data: *const c_double,
    rows: usize,
    cols: usize,
    y: *const c_double,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        let x = input(data, rows, cols)?;
        let t = targets(y, rows)?;
        let mut m = RandomForestClassifier::new(h.n_estimators, h.max_depth)
            .with_min_samples_leaf(h.min_samples_leaf)
            .with_min_samples_split(h.min_samples_split)
            .with_max_leaf_nodes(optional_leaves(h.max_leaf_nodes))
            .with_criterion(h.criterion)
            .with_seed(h.seed);
        if let Some(f) = optional_features(h.max_features) {
            m = m.with_max_features(f);
        }
        SupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols)?, t)?;
        h.fitted = Some(m);
        Ok(())
    })
}

/// Predicts class labels for a row-major `rows × cols` matrix. `out` receives `rows` values.
///
/// # Safety
/// `data` is valid for `rows * cols` reads and `out` for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_random_forest_classifier_predict(
    h: *const TpmlRandomForestClassifier,
    data: *const c_double,
    rows: usize,
    cols: usize,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h
            .fitted
            .as_ref()
            .ok_or(tpt_ml_proven_core::error::Error::NotFitted)?;
        let x = input(data, rows, cols)?;
        let p = Predictor::predict(m, DenseView::new(x, rows, cols)?)?;
        output(out, out_len, &p)
    })
}

/// Fraction of rows of a row-major matrix whose predicted label equals the label in `y`.
///
/// # Safety
/// `data` is valid for `rows * cols` reads, and `y` for `rows` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_random_forest_classifier_score(
    h: *const TpmlRandomForestClassifier,
    data: *const c_double,
    rows: usize,
    cols: usize,
    y: *const c_double,
    out: *mut c_double,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h
            .fitted
            .as_ref()
            .ok_or(tpt_ml_proven_core::error::Error::NotFitted)?;
        let x = input(data, rows, cols)?;
        let t = targets(y, rows)?;
        let p = Predictor::predict(m, DenseView::new(x, rows, cols)?)?;
        let score = accuracy(t, &p).map_err(Failure::from)?;
        output(out, 1, &[score])
    })
}

// ---------------------------------------------------------------- settings and importances

/// Checks and stores a minimum split size. It must be at least 2.
fn set_min_split(current: &mut usize, n: usize) -> Result<(), Failure> {
    if n < 2 {
        return Err(Failure::new(
            TpmlStatus::InvalidArgument,
            "min_samples_split must be at least 2",
        ));
    }
    *current = n;
    Ok(())
}

/// Checks and stores a leaf limit. Zero means no limit; otherwise it must be at least 2.
fn set_leaf_limit(current: &mut usize, n: usize) -> Result<(), Failure> {
    if n == 1 {
        return Err(Failure::new(
            TpmlStatus::InvalidArgument,
            "max_leaf_nodes must be at least 2, or 0 for no limit",
        ));
    }
    *current = n;
    Ok(())
}

/// Writes a model's feature importances, or reports `NotFitted` when it has none.
///
/// # Safety
/// `out` must be valid for `out_len` `f64` writes.
unsafe fn importances_into(
    importances: Option<Vec<f64>>,
    out: *mut c_double,
    out_len: usize,
) -> Result<(), Failure> {
    let values = importances.ok_or(tpt_ml_proven_core::error::Error::NotFitted)?;
    output(out, out_len, &values)
}

/// Sets the fewest rows a node needs before it may be split in a decision tree regressor
/// (default 2). Call before `fit`.
///
/// # Safety
/// `h` must be a live handle.
#[no_mangle]
pub unsafe extern "C" fn tpml_decision_tree_regressor_set_min_samples_split(
    h: *mut TpmlDecisionTreeRegressor,
    n: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        set_min_split(&mut h.min_samples_split, n)
    })
}

/// Limits a decision tree regressor to `n` leaves, grown best-first. Zero means no limit
/// (the default). Call before `fit`.
///
/// # Safety
/// `h` must be a live handle.
#[no_mangle]
pub unsafe extern "C" fn tpml_decision_tree_regressor_set_max_leaf_nodes(
    h: *mut TpmlDecisionTreeRegressor,
    n: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        set_leaf_limit(&mut h.max_leaf_nodes, n)
    })
}

/// Writes each feature's share of the impurity reduction into `out`, which holds one value per
/// feature. Reports `NotFitted` before `fit`.
///
/// # Safety
/// `h` must be a live handle and `out` valid for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_decision_tree_regressor_feature_importances(
    h: *const TpmlDecisionTreeRegressor,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let imp = h.fitted.as_ref().and_then(|m| m.feature_importances());
        importances_into(imp, out, out_len)
    })
}

/// Sets the fewest rows a node needs before it may be split in a decision tree classifier
/// (default 2). Call before `fit`.
///
/// # Safety
/// `h` must be a live handle.
#[no_mangle]
pub unsafe extern "C" fn tpml_decision_tree_classifier_set_min_samples_split(
    h: *mut TpmlDecisionTreeClassifier,
    n: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        set_min_split(&mut h.min_samples_split, n)
    })
}

/// Limits a decision tree classifier to `n` leaves, grown best-first. Zero means no limit (the
/// default). Call before `fit`.
///
/// # Safety
/// `h` must be a live handle.
#[no_mangle]
pub unsafe extern "C" fn tpml_decision_tree_classifier_set_max_leaf_nodes(
    h: *mut TpmlDecisionTreeClassifier,
    n: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        set_leaf_limit(&mut h.max_leaf_nodes, n)
    })
}

/// Writes each feature's share of the impurity reduction into `out`, which holds one value per
/// feature. Reports `NotFitted` before `fit`.
///
/// # Safety
/// `h` must be a live handle and `out` valid for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_decision_tree_classifier_feature_importances(
    h: *const TpmlDecisionTreeClassifier,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let imp = h.fitted.as_ref().and_then(|m| m.feature_importances());
        importances_into(imp, out, out_len)
    })
}

/// Sets the fewest rows a node needs before it may be split in each boosting tree (default 2).
/// Call before `fit`.
///
/// # Safety
/// `h` must be a live handle.
#[no_mangle]
pub unsafe extern "C" fn tpml_gradient_boosting_set_min_samples_split(
    h: *mut TpmlGradientBoosting,
    n: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        set_min_split(&mut h.min_samples_split, n)
    })
}

/// Limits each boosting tree to `n` leaves, grown best-first. Zero means no limit (the default).
/// Call before `fit`.
///
/// # Safety
/// `h` must be a live handle.
#[no_mangle]
pub unsafe extern "C" fn tpml_gradient_boosting_set_max_leaf_nodes(
    h: *mut TpmlGradientBoosting,
    n: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        set_leaf_limit(&mut h.max_leaf_nodes, n)
    })
}

/// Writes each feature's share of the squared-error reduction over all rounds into `out`, which
/// holds one value per feature. Reports `NotFitted` before `fit`.
///
/// # Safety
/// `h` must be a live handle and `out` valid for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_gradient_boosting_feature_importances(
    h: *const TpmlGradientBoosting,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let imp = h.fitted.as_ref().and_then(|m| m.feature_importances());
        importances_into(imp, out, out_len)
    })
}

/// Sets the fewest rows a node needs before it may be split in each forest tree (default 2).
/// Call before `fit`.
///
/// # Safety
/// `h` must be a live handle.
#[no_mangle]
pub unsafe extern "C" fn tpml_random_forest_regressor_set_min_samples_split(
    h: *mut TpmlRandomForestRegressor,
    n: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        set_min_split(&mut h.min_samples_split, n)
    })
}

/// Limits each forest tree to `n` leaves, grown best-first. Zero means no limit (the default).
/// Call before `fit`.
///
/// # Safety
/// `h` must be a live handle.
#[no_mangle]
pub unsafe extern "C" fn tpml_random_forest_regressor_set_max_leaf_nodes(
    h: *mut TpmlRandomForestRegressor,
    n: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        set_leaf_limit(&mut h.max_leaf_nodes, n)
    })
}

/// Writes each feature's average share of the impurity reduction into `out`, which holds one
/// value per feature. Reports `NotFitted` before `fit`.
///
/// # Safety
/// `h` must be a live handle and `out` valid for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_random_forest_regressor_feature_importances(
    h: *const TpmlRandomForestRegressor,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let imp = h.fitted.as_ref().and_then(|m| m.feature_importances());
        importances_into(imp, out, out_len)
    })
}

/// Sets the fewest rows a node needs before it may be split in each forest tree (default 2).
/// Call before `fit`.
///
/// # Safety
/// `h` must be a live handle.
#[no_mangle]
pub unsafe extern "C" fn tpml_random_forest_classifier_set_min_samples_split(
    h: *mut TpmlRandomForestClassifier,
    n: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        set_min_split(&mut h.min_samples_split, n)
    })
}

/// Limits each forest tree to `n` leaves, grown best-first. Zero means no limit (the default).
/// Call before `fit`.
///
/// # Safety
/// `h` must be a live handle.
#[no_mangle]
pub unsafe extern "C" fn tpml_random_forest_classifier_set_max_leaf_nodes(
    h: *mut TpmlRandomForestClassifier,
    n: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        set_leaf_limit(&mut h.max_leaf_nodes, n)
    })
}

/// Writes each feature's average share of the impurity reduction into `out`, which holds one
/// value per feature. Reports `NotFitted` before `fit`.
///
/// # Safety
/// `h` must be a live handle and `out` valid for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_random_forest_classifier_feature_importances(
    h: *const TpmlRandomForestClassifier,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let imp = h.fitted.as_ref().and_then(|m| m.feature_importances());
        importances_into(imp, out, out_len)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_regression_tree_fits_a_step_through_the_c_calls() {
        let x: Vec<f64> = (0..8).map(f64::from).collect();
        let y = [1.0, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0];
        let h = tpml_decision_tree_regressor_new(2, 1);
        assert!(!h.is_null());
        let mut out = vec![0.0; 8];
        unsafe {
            assert_eq!(
                tpml_decision_tree_regressor_fit(h, x.as_ptr(), 8, 1, y.as_ptr()),
                TpmlStatus::Ok
            );
            assert_eq!(
                tpml_decision_tree_regressor_predict(
                    h,
                    x.as_ptr(),
                    8,
                    1,
                    out.as_mut_ptr(),
                    out.len()
                ),
                TpmlStatus::Ok
            );
            tpml_decision_tree_regressor_free(h);
        }
        assert_eq!(out, y.to_vec());
    }

    #[test]
    fn a_classifier_reports_its_class_count() {
        let x: Vec<f64> = (0..8).map(f64::from).collect();
        let y = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let h = tpml_decision_tree_classifier_new(2, 1, TpmlCriterion::Entropy);
        let mut classes = 0usize;
        unsafe {
            assert_eq!(
                tpml_decision_tree_classifier_fit(h, x.as_ptr(), 8, 1, y.as_ptr()),
                TpmlStatus::Ok
            );
            assert_eq!(
                tpml_decision_tree_classifier_n_classes(h, &mut classes),
                TpmlStatus::Ok
            );
            tpml_decision_tree_classifier_free(h);
        }
        assert_eq!(classes, 2);
    }

    #[test]
    fn invalid_settings_give_a_null_handle() {
        assert!(tpml_decision_tree_regressor_new(0, 1).is_null());
        assert!(tpml_gradient_boosting_new(10, 0.0, 2, 1, 1.0, 0, 0).is_null());
        assert!(tpml_gradient_boosting_new(10, 0.1, 2, 1, 1.5, 0, 0).is_null());
        assert!(tpml_random_forest_classifier_new(0, 2, 1, TpmlCriterion::Gini, 0, 0).is_null());
    }

    #[test]
    fn a_forest_predicts_its_training_labels() {
        let x: Vec<f64> = (0..20).map(f64::from).collect();
        let y: Vec<f64> = (0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }).collect();
        let h = tpml_random_forest_classifier_new(9, 3, 1, TpmlCriterion::Gini, 0, 3);
        let mut out = vec![0.0; 20];
        let mut acc = 0.0;
        unsafe {
            assert_eq!(
                tpml_random_forest_classifier_fit(h, x.as_ptr(), 20, 1, y.as_ptr()),
                TpmlStatus::Ok
            );
            assert_eq!(
                tpml_random_forest_classifier_predict(
                    h,
                    x.as_ptr(),
                    20,
                    1,
                    out.as_mut_ptr(),
                    out.len()
                ),
                TpmlStatus::Ok
            );
            assert_eq!(
                tpml_random_forest_classifier_score(h, x.as_ptr(), 20, 1, y.as_ptr(), &mut acc),
                TpmlStatus::Ok
            );
            tpml_random_forest_classifier_free(h);
        }
        assert_eq!(out, y);
        assert_eq!(acc, 1.0);
    }
}

#[cfg(test)]
mod settings_tests {
    use super::*;

    #[test]
    fn settings_are_validated_and_importances_need_a_fit() {
        unsafe {
            let h = tpml_decision_tree_regressor_new(3, 1);
            assert!(!h.is_null());
            assert_eq!(
                tpml_decision_tree_regressor_set_min_samples_split(h, 1),
                TpmlStatus::InvalidArgument
            );
            assert_eq!(
                tpml_decision_tree_regressor_set_min_samples_split(h, 6),
                TpmlStatus::Ok
            );
            assert_eq!(
                tpml_decision_tree_regressor_set_max_leaf_nodes(h, 1),
                TpmlStatus::InvalidArgument
            );
            assert_eq!(
                tpml_decision_tree_regressor_set_max_leaf_nodes(h, 0),
                TpmlStatus::Ok
            );
            let mut imp = [0.0_f64; 2];
            assert_eq!(
                tpml_decision_tree_regressor_feature_importances(h, imp.as_mut_ptr(), 2),
                TpmlStatus::NotFitted
            );
            tpml_decision_tree_regressor_free(h);
        }
    }

    #[test]
    fn a_fitted_forest_reports_importances_of_the_feature_count() {
        let data: Vec<f64> = (0..12).flat_map(|i| [i as f64, 2.0]).collect();
        let y: Vec<f64> = (0..12).map(|i| if i < 6 { 0.0 } else { 1.0 }).collect();
        unsafe {
            let h = tpml_random_forest_regressor_new(5, 3, 1, 0, 1);
            assert_eq!(
                tpml_random_forest_regressor_set_max_leaf_nodes(h, 3),
                TpmlStatus::Ok
            );
            assert_eq!(
                tpml_random_forest_regressor_fit(h, data.as_ptr(), 12, 2, y.as_ptr()),
                TpmlStatus::Ok
            );
            let mut imp = [0.0_f64; 2];
            assert_eq!(
                tpml_random_forest_regressor_feature_importances(h, imp.as_mut_ptr(), 2),
                TpmlStatus::Ok
            );
            assert!((imp[0] + imp[1] - 1.0).abs() < 1e-12);
            assert_eq!(imp[1], 0.0);
            let mut wrong = [0.0_f64; 3];
            assert_eq!(
                tpml_random_forest_regressor_feature_importances(h, wrong.as_mut_ptr(), 3),
                TpmlStatus::InvalidArgument
            );
            tpml_random_forest_regressor_free(h);
        }
    }
}
