// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! C ABI for the Phase 8 estimators, in `f64`. Same conventions as the rest of the ABI: a
//! `TpmlStatus` return, caller-allocated outputs with their element counts, and opaque
//! handles freed with their `*_free` function.
//!

use std::ffi::c_double;

use tpt_ml_proven_clustering::{Dbscan, Hdbscan};
use tpt_ml_proven_core::error::Error;
use tpt_ml_proven_core::ArrayView;
use tpt_ml_proven_core::{
    DenseView, Predictor, SupervisedEstimator, Transformer, UnsupervisedEstimator,
};
use tpt_ml_proven_decomposition::{Nmf, TruncatedSvd};
use tpt_ml_proven_ensemble::StumpForest;
use tpt_ml_proven_linear::{ElasticNet, Lasso, LogisticRegression, MulticlassLogistic, Ridge};

use super::{input, output, run, Failure, TpmlStatus};

/// The handle is null.
pub(crate) fn null_handle() -> Failure {
    Failure::new(TpmlStatus::NullPointer, "handle is null")
}

/// Borrows one target per row: `rows` values.
///
/// # Safety
/// `y` must be valid for reads of `rows` `f64` values.
pub(crate) unsafe fn targets<'a>(y: *const f64, rows: usize) -> Result<&'a [f64], Failure> {
    input(y, rows, 1)
}

/// Writes one matrix of `rows × cols` values into a caller buffer of `out_len` elements.
fn flat_output(
    out: *mut c_double,
    out_len: usize,
    m: &tpt_ml_proven_core::Dense<f64>,
) -> Result<(), Failure> {
    let mut flat = Vec::with_capacity(m.rows() * m.cols());
    for r in 0..m.rows() {
        for c in 0..m.cols() {
            flat.push(m.get(r, c));
        }
    }
    // SAFETY: the caller promises `out` is valid for `out_len` writes; `output` checks the length.
    unsafe { output(out, out_len, &flat) }
}

// ---------------------------------------------------------------- DBSCAN

/// Opaque DBSCAN handle.
pub struct TpmlDbscan {
    eps: f64,
    min_samples: usize,
    fitted: Option<Dbscan<f64>>,
}

/// Creates a DBSCAN estimator. Free it with `tpml_dbscan_free`. Returns null if `eps` is not
/// positive or `min_samples` is zero.
#[no_mangle]
pub extern "C" fn tpml_dbscan_new(eps: f64, min_samples: usize) -> *mut TpmlDbscan {
    if !(eps.is_finite() && eps > 0.0) || min_samples == 0 {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(TpmlDbscan {
        eps,
        min_samples,
        fitted: None,
    }))
}

/// Frees a DBSCAN handle. Null is ignored.
///
/// # Safety
/// `h` must be null or a handle from `tpml_dbscan_new` that has not been freed.
#[no_mangle]
pub unsafe extern "C" fn tpml_dbscan_free(h: *mut TpmlDbscan) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Fits to a row-major `rows × cols` matrix.
///
/// # Safety
/// `h` must be a live handle and `data` valid for `rows * cols` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_dbscan_fit(
    h: *mut TpmlDbscan,
    data: *const c_double,
    rows: usize,
    cols: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        let x = input(data, rows, cols)?;
        let mut m = Dbscan::<f64>::new(h.eps, h.min_samples);
        UnsupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols)?)?;
        h.fitted = Some(m);
        Ok(())
    })
}

/// Writes the label of each training row (`-1` for noise) into `out`, which holds `out_len`
/// values.
///
/// # Safety
/// `h` must be a live handle and `out` valid for `out_len` `i64` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_dbscan_labels(
    h: *const TpmlDbscan,
    out: *mut i64,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        output(out, out_len, m.labels().unwrap_or(&[]))
    })
}

// ---------------------------------------------------------------- HDBSCAN

/// Opaque HDBSCAN handle.
pub struct TpmlHdbscan {
    min_cluster_size: usize,
    min_samples: usize,
    fitted: Option<Hdbscan<f64>>,
}

/// Creates an HDBSCAN estimator. `min_samples` of zero means "same as `min_cluster_size`".
/// Free it with `tpml_hdbscan_free`. Returns null if `min_cluster_size` is below 2.
#[no_mangle]
pub extern "C" fn tpml_hdbscan_new(
    min_cluster_size: usize,
    min_samples: usize,
) -> *mut TpmlHdbscan {
    if min_cluster_size < 2 {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(TpmlHdbscan {
        min_cluster_size,
        min_samples,
        fitted: None,
    }))
}

/// Frees an HDBSCAN handle. Null is ignored.
///
/// # Safety
/// `h` must be null or a handle from `tpml_hdbscan_new` that has not been freed.
#[no_mangle]
pub unsafe extern "C" fn tpml_hdbscan_free(h: *mut TpmlHdbscan) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Fits to a row-major `rows × cols` matrix.
///
/// # Safety
/// `h` must be a live handle and `data` valid for `rows * cols` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_hdbscan_fit(
    h: *mut TpmlHdbscan,
    data: *const c_double,
    rows: usize,
    cols: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        let x = input(data, rows, cols)?;
        let mut m = Hdbscan::<f64>::new(h.min_cluster_size);
        if h.min_samples > 0 {
            m = m.with_min_samples(h.min_samples);
        }
        UnsupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols)?)?;
        h.fitted = Some(m);
        Ok(())
    })
}

/// Writes the label of each training row (`-1` for noise) into `out`.
///
/// # Safety
/// `h` must be a live handle and `out` valid for `out_len` `i64` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_hdbscan_labels(
    h: *const TpmlHdbscan,
    out: *mut i64,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        output(out, out_len, m.labels().unwrap_or(&[]))
    })
}

// ---------------------------------------------------------------- TruncatedSVD

/// Opaque TruncatedSVD handle.
pub struct TpmlTruncatedSvd {
    n_components: usize,
    fitted: Option<TruncatedSvd<f64>>,
}

/// Creates a TruncatedSVD estimator keeping `n_components` vectors. Free it with
/// `tpml_truncated_svd_free`. Returns null if `n_components` is zero.
#[no_mangle]
pub extern "C" fn tpml_truncated_svd_new(n_components: usize) -> *mut TpmlTruncatedSvd {
    if n_components == 0 {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(TpmlTruncatedSvd {
        n_components,
        fitted: None,
    }))
}

/// Frees a TruncatedSVD handle. Null is ignored.
///
/// # Safety
/// `h` must be null or a handle from `tpml_truncated_svd_new` that has not been freed.
#[no_mangle]
pub unsafe extern "C" fn tpml_truncated_svd_free(h: *mut TpmlTruncatedSvd) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Fits to a row-major `rows × cols` matrix.
///
/// # Safety
/// `h` must be a live handle and `data` valid for `rows * cols` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_truncated_svd_fit(
    h: *mut TpmlTruncatedSvd,
    data: *const c_double,
    rows: usize,
    cols: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        let x = input(data, rows, cols)?;
        let mut m = TruncatedSvd::<f64>::new(h.n_components);
        UnsupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols)?)?;
        h.fitted = Some(m);
        Ok(())
    })
}

/// Projects a row-major `rows × cols` matrix. `out` receives `rows × n_components` values.
///
/// # Safety
/// `h` must be a live handle, `data` valid for `rows * cols` reads, and `out` valid for
/// `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_truncated_svd_transform(
    h: *const TpmlTruncatedSvd,
    data: *const c_double,
    rows: usize,
    cols: usize,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        let x = input(data, rows, cols)?;
        let z = m.transform(DenseView::new(x, rows, cols)?)?;
        flat_output(out, out_len, &z)
    })
}

/// Writes the singular values, largest first, into `out` (`n_components` values).
///
/// # Safety
/// `h` must be a live handle and `out` valid for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_truncated_svd_singular_values(
    h: *const TpmlTruncatedSvd,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        output(out, out_len, m.singular_values().unwrap_or(&[]))
    })
}

// ---------------------------------------------------------------- NMF

/// Opaque NMF handle.
pub struct TpmlNmf {
    n_components: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    fitted: Option<Nmf<f64>>,
}

/// Creates an NMF estimator. Free it with `tpml_nmf_free`. Returns null if `n_components`
/// is zero.
#[no_mangle]
pub extern "C" fn tpml_nmf_new(
    n_components: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> *mut TpmlNmf {
    if n_components == 0 || !tol.is_finite() {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(TpmlNmf {
        n_components,
        max_iter,
        tol,
        seed,
        fitted: None,
    }))
}

/// Frees an NMF handle. Null is ignored.
///
/// # Safety
/// `h` must be null or a handle from `tpml_nmf_new` that has not been freed.
#[no_mangle]
pub unsafe extern "C" fn tpml_nmf_free(h: *mut TpmlNmf) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Fits to a non-negative row-major `rows × cols` matrix.
///
/// # Safety
/// `h` must be a live handle and `data` valid for `rows * cols` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_nmf_fit(
    h: *mut TpmlNmf,
    data: *const c_double,
    rows: usize,
    cols: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        let x = input(data, rows, cols)?;
        let mut m = Nmf::<f64>::new(h.n_components)
            .with_max_iter(h.max_iter)
            .with_tol(h.tol)
            .with_seed(h.seed);
        UnsupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols)?)?;
        h.fitted = Some(m);
        Ok(())
    })
}

/// Writes the factor `W` for a row-major `rows × cols` matrix into `out`
/// (`rows × n_components` values).
///
/// # Safety
/// `h` must be a live handle, `data` valid for `rows * cols` reads, and `out` valid for
/// `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_nmf_transform(
    h: *const TpmlNmf,
    data: *const c_double,
    rows: usize,
    cols: usize,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        let x = input(data, rows, cols)?;
        let w = m.transform(DenseView::new(x, rows, cols)?)?;
        flat_output(out, out_len, &w)
    })
}

/// Writes the learned factors `H` (`n_components × cols`, row-major) into `out`.
///
/// # Safety
/// `h` must be a live handle and `out` valid for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_nmf_components(
    h: *const TpmlNmf,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        output(
            out,
            out_len,
            m.components().map_or(&[][..], |d| d.as_slice()),
        )
    })
}

// ---------------------------------------------------------------- Ridge

/// Opaque Ridge handle.
pub struct TpmlRidge {
    alpha: f64,
    fitted: Option<Ridge<f64>>,
}

/// Creates a Ridge regression with penalty `alpha`. Free it with `tpml_ridge_free`. Returns
/// null if `alpha` is negative or not finite.
#[no_mangle]
pub extern "C" fn tpml_ridge_new(alpha: f64) -> *mut TpmlRidge {
    if !(alpha.is_finite() && alpha >= 0.0) {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(TpmlRidge {
        alpha,
        fitted: None,
    }))
}

/// Frees a Ridge handle. Null is ignored.
///
/// # Safety
/// `h` must be null or a handle from `tpml_ridge_new` that has not been freed.
#[no_mangle]
pub unsafe extern "C" fn tpml_ridge_free(h: *mut TpmlRidge) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Fits to a row-major `rows × cols` matrix and `rows` targets.
///
/// # Safety
/// `h` must be a live handle, `data` valid for `rows * cols` reads, and `y` for `rows` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_ridge_fit(
    h: *mut TpmlRidge,
    data: *const c_double,
    rows: usize,
    cols: usize,
    y: *const c_double,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        let x = input(data, rows, cols)?;
        let t = targets(y, rows)?;
        let mut m = Ridge::<f64>::new(h.alpha);
        SupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols)?, t)?;
        h.fitted = Some(m);
        Ok(())
    })
}

/// Predicts for a row-major `rows × cols` matrix. `out` receives `rows` values.
///
/// # Safety
/// `h` must be a live handle, `data` valid for `rows * cols` reads, and `out` valid for
/// `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_ridge_predict(
    h: *const TpmlRidge,
    data: *const c_double,
    rows: usize,
    cols: usize,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        let x = input(data, rows, cols)?;
        let p = Predictor::predict(m, DenseView::new(x, rows, cols)?)?;
        output(out, out_len, &p)
    })
}

/// Writes the coefficients (`cols` values) into `out`.
///
/// # Safety
/// `h` must be a live handle and `out` valid for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_ridge_coef(
    h: *const TpmlRidge,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        output(out, out_len, m.coef().unwrap_or(&[]))
    })
}

/// Writes the intercept into `out`.
///
/// # Safety
/// `h` must be a live handle and `out` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn tpml_ridge_intercept(
    h: *const TpmlRidge,
    out: *mut c_double,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        let b = m.intercept().ok_or(Error::NotFitted)?;
        output(out, 1, &[b])
    })
}

// ---------------------------------------------------------------- Lasso and ElasticNet

/// Opaque Lasso handle.
pub struct TpmlLasso {
    alpha: f64,
    max_iter: usize,
    tol: f64,
    fitted: Option<Lasso<f64>>,
}

/// Creates a Lasso with penalty `alpha`. Free it with `tpml_lasso_free`. Returns null if
/// `alpha` is negative or not finite, or `tol` is not finite.
#[no_mangle]
pub extern "C" fn tpml_lasso_new(alpha: f64, max_iter: usize, tol: f64) -> *mut TpmlLasso {
    if !(alpha.is_finite() && alpha >= 0.0) || !tol.is_finite() {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(TpmlLasso {
        alpha,
        max_iter,
        tol,
        fitted: None,
    }))
}

/// Frees a Lasso handle. Null is ignored.
///
/// # Safety
/// `h` must be null or a handle from `tpml_lasso_new` that has not been freed.
#[no_mangle]
pub unsafe extern "C" fn tpml_lasso_free(h: *mut TpmlLasso) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Fits to a row-major `rows × cols` matrix and `rows` targets.
///
/// # Safety
/// `h` must be a live handle, `data` valid for `rows * cols` reads, and `y` for `rows` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_lasso_fit(
    h: *mut TpmlLasso,
    data: *const c_double,
    rows: usize,
    cols: usize,
    y: *const c_double,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        let x = input(data, rows, cols)?;
        let t = targets(y, rows)?;
        let mut m = Lasso::<f64>::new(h.alpha)
            .with_max_iter(h.max_iter)
            .with_tol(h.tol);
        SupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols)?, t)?;
        h.fitted = Some(m);
        Ok(())
    })
}

/// Predicts for a row-major `rows × cols` matrix. `out` receives `rows` values.
///
/// # Safety
/// `h` must be a live handle, `data` valid for `rows * cols` reads, and `out` valid for
/// `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_lasso_predict(
    h: *const TpmlLasso,
    data: *const c_double,
    rows: usize,
    cols: usize,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        let x = input(data, rows, cols)?;
        let p = Predictor::predict(m, DenseView::new(x, rows, cols)?)?;
        output(out, out_len, &p)
    })
}

/// Writes the coefficients (`cols` values) into `out`.
///
/// # Safety
/// `h` must be a live handle and `out` valid for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_lasso_coef(
    h: *const TpmlLasso,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        output(out, out_len, m.coef().unwrap_or(&[]))
    })
}

/// Writes the intercept into `out`.
///
/// # Safety
/// `h` must be a live handle and `out` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn tpml_lasso_intercept(
    h: *const TpmlLasso,
    out: *mut c_double,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        let b = m.intercept().ok_or(Error::NotFitted)?;
        output(out, 1, &[b])
    })
}

/// Opaque ElasticNet handle.
pub struct TpmlElasticNet {
    alpha: f64,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
    fitted: Option<ElasticNet<f64>>,
}

/// Creates an ElasticNet with penalty `alpha` and mixing `l1_ratio` in `[0, 1]`. Free it with
/// `tpml_elastic_net_free`. Returns null for invalid parameters.
#[no_mangle]
pub extern "C" fn tpml_elastic_net_new(
    alpha: f64,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
) -> *mut TpmlElasticNet {
    if !(alpha.is_finite() && alpha >= 0.0) || !(0.0..=1.0).contains(&l1_ratio) || !tol.is_finite()
    {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(TpmlElasticNet {
        alpha,
        l1_ratio,
        max_iter,
        tol,
        fitted: None,
    }))
}

/// Frees an ElasticNet handle. Null is ignored.
///
/// # Safety
/// `h` must be null or a handle from `tpml_elastic_net_new` that has not been freed.
#[no_mangle]
pub unsafe extern "C" fn tpml_elastic_net_free(h: *mut TpmlElasticNet) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Fits to a row-major `rows × cols` matrix and `rows` targets.
///
/// # Safety
/// `h` must be a live handle, `data` valid for `rows * cols` reads, and `y` for `rows` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_elastic_net_fit(
    h: *mut TpmlElasticNet,
    data: *const c_double,
    rows: usize,
    cols: usize,
    y: *const c_double,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        let x = input(data, rows, cols)?;
        let t = targets(y, rows)?;
        let mut m = ElasticNet::<f64>::new(h.alpha, h.l1_ratio)
            .with_max_iter(h.max_iter)
            .with_tol(h.tol);
        SupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols)?, t)?;
        h.fitted = Some(m);
        Ok(())
    })
}

/// Predicts for a row-major `rows × cols` matrix. `out` receives `rows` values.
///
/// # Safety
/// `h` must be a live handle, `data` valid for `rows * cols` reads, and `out` valid for
/// `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_elastic_net_predict(
    h: *const TpmlElasticNet,
    data: *const c_double,
    rows: usize,
    cols: usize,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        let x = input(data, rows, cols)?;
        let p = Predictor::predict(m, DenseView::new(x, rows, cols)?)?;
        output(out, out_len, &p)
    })
}

/// Writes the coefficients (`cols` values) into `out`.
///
/// # Safety
/// `h` must be a live handle and `out` valid for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_elastic_net_coef(
    h: *const TpmlElasticNet,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        output(out, out_len, m.coef().unwrap_or(&[]))
    })
}

/// Writes the intercept into `out`.
///
/// # Safety
/// `h` must be a live handle and `out` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn tpml_elastic_net_intercept(
    h: *const TpmlElasticNet,
    out: *mut c_double,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        let b = m.intercept().ok_or(Error::NotFitted)?;
        output(out, 1, &[b])
    })
}

// ---------------------------------------------------------------- logistic regression

enum LogitState {
    Binary {
        model: LogisticRegression<f64>,
        classes: [f64; 2],
    },
    Multi(MulticlassLogistic<f64>),
}

/// Opaque logistic regression handle.
pub struct TpmlLogistic {
    c: f64,
    max_iter: usize,
    tol: f64,
    fitted: Option<LogitState>,
}

/// Creates a logistic regression with inverse regularisation strength `c`. Free it with
/// `tpml_logistic_free`. Returns null if `c` is not positive.
#[no_mangle]
pub extern "C" fn tpml_logistic_new(c: f64, max_iter: usize, tol: f64) -> *mut TpmlLogistic {
    if !(c.is_finite() && c > 0.0) {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(TpmlLogistic {
        c,
        max_iter,
        tol,
        fitted: None,
    }))
}

/// Frees a logistic regression handle. Null is ignored.
///
/// # Safety
/// `h` must be null or a handle from `tpml_logistic_new` that has not been freed.
#[no_mangle]
pub unsafe extern "C" fn tpml_logistic_free(h: *mut TpmlLogistic) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Fits to a row-major `rows × cols` matrix and `rows` class labels. Two distinct labels
/// give a binary model; more give a multinomial one.
///
/// # Safety
/// `h` must be a live handle, `data` valid for `rows * cols` reads, and `y` for `rows` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_logistic_fit(
    h: *mut TpmlLogistic,
    data: *const c_double,
    rows: usize,
    cols: usize,
    y: *const c_double,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        let x = input(data, rows, cols)?;
        let t = targets(y, rows)?;
        let mut classes = t.to_vec();
        classes.sort_by(f64::total_cmp);
        classes.dedup();
        let state = if classes.len() == 2 {
            let y01: Vec<f64> = t
                .iter()
                .map(|&v| if v == classes[1] { 1.0 } else { 0.0 })
                .collect();
            let mut m = LogisticRegression::<f64>::new(h.c)
                .with_max_iter(h.max_iter)
                .with_tol(h.tol);
            SupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols)?, &y01[..])?;
            LogitState::Binary {
                model: m,
                classes: [classes[0], classes[1]],
            }
        } else {
            let mut m = MulticlassLogistic::<f64>::new(h.c)
                .with_max_iter(h.max_iter)
                .with_tol(h.tol);
            SupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols)?, t)?;
            LogitState::Multi(m)
        };
        h.fitted = Some(state);
        Ok(())
    })
}

/// Writes the number of classes seen in training into `out`.
///
/// # Safety
/// `h` must be a live handle and `out` valid for one `usize` write.
#[no_mangle]
pub unsafe extern "C" fn tpml_logistic_n_classes(
    h: *const TpmlLogistic,
    out: *mut usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let n = match h.fitted.as_ref().ok_or(Error::NotFitted)? {
            LogitState::Binary { .. } => 2,
            LogitState::Multi(m) => m.classes().map_or(0, <[f64]>::len),
        };
        output(out, 1, &[n])
    })
}

/// Writes the predicted class labels for a row-major `rows × cols` matrix into `out`.
///
/// # Safety
/// `h` must be a live handle, `data` valid for `rows * cols` reads, and `out` valid for
/// `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_logistic_predict(
    h: *const TpmlLogistic,
    data: *const c_double,
    rows: usize,
    cols: usize,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let state = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        let x = input(data, rows, cols)?;
        let labels: Vec<f64> = match state {
            LogitState::Binary { model, classes } => {
                Predictor::predict(model, DenseView::new(x, rows, cols)?)?
                    .into_iter()
                    .map(|z| if z == 1.0 { classes[1] } else { classes[0] })
                    .collect()
            }
            LogitState::Multi(model) => Predictor::predict(model, DenseView::new(x, rows, cols)?)?,
        };
        output(out, out_len, &labels)
    })
}

/// Writes class probabilities for a row-major `rows × cols` matrix into `out`
/// (`rows × n_classes` values, classes in ascending order).
///
/// # Safety
/// `h` must be a live handle, `data` valid for `rows * cols` reads, and `out` valid for
/// `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_logistic_predict_proba(
    h: *const TpmlLogistic,
    data: *const c_double,
    rows: usize,
    cols: usize,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let state = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        let x = input(data, rows, cols)?;
        let flat: Vec<f64> = match state {
            LogitState::Binary { model, .. } => model
                .predict_proba(DenseView::new(x, rows, cols)?)?
                .into_iter()
                .flat_map(|p1| [1.0 - p1, p1])
                .collect(),
            LogitState::Multi(model) => model.predict_proba(DenseView::new(x, rows, cols)?)?,
        };
        output(out, out_len, &flat)
    })
}

// ---------------------------------------------------------------- stump forest

/// Opaque stump forest handle.
pub struct TpmlStumpForest {
    n_estimators: usize,
    seed: u64,
    fitted: Option<StumpForest<f64>>,
}

/// Creates a stump forest of `n_estimators` stumps. Free it with `tpml_stump_forest_free`.
/// Returns null if `n_estimators` is zero.
#[no_mangle]
pub extern "C" fn tpml_stump_forest_new(n_estimators: usize, seed: u64) -> *mut TpmlStumpForest {
    if n_estimators == 0 {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(TpmlStumpForest {
        n_estimators,
        seed,
        fitted: None,
    }))
}

/// Frees a stump forest handle. Null is ignored.
///
/// # Safety
/// `h` must be null or a handle from `tpml_stump_forest_new` that has not been freed.
#[no_mangle]
pub unsafe extern "C" fn tpml_stump_forest_free(h: *mut TpmlStumpForest) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Fits to a row-major `rows × cols` matrix and `rows` class labels.
///
/// # Safety
/// `h` must be a live handle, `data` valid for `rows * cols` reads, and `y` for `rows` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_stump_forest_fit(
    h: *mut TpmlStumpForest,
    data: *const c_double,
    rows: usize,
    cols: usize,
    y: *const c_double,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        let x = input(data, rows, cols)?;
        let t = targets(y, rows)?;
        let mut m = StumpForest::<f64>::new(h.n_estimators).with_seed(h.seed);
        SupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols)?, t)?;
        h.fitted = Some(m);
        Ok(())
    })
}

/// Predicts class labels for a row-major `rows × cols` matrix. `out` receives `rows` values.
///
/// # Safety
/// `h` must be a live handle, `data` valid for `rows * cols` reads, and `out` valid for
/// `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_stump_forest_predict(
    h: *const TpmlStumpForest,
    data: *const c_double,
    rows: usize,
    cols: usize,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        let x = input(data, rows, cols)?;
        let p = Predictor::predict(m, DenseView::new(x, rows, cols)?)?;
        output(out, out_len, &p)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dbscan_labels_through_the_c_abi() {
        let data = [
            0.0_f64, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
        ];
        unsafe {
            let h = tpml_dbscan_new(1.0, 3);
            assert!(!h.is_null());
            assert_eq!(tpml_dbscan_fit(h, data.as_ptr(), 6, 2), TpmlStatus::Ok);
            let mut labels = [0_i64; 6];
            assert_eq!(
                tpml_dbscan_labels(h, labels.as_mut_ptr(), 6),
                TpmlStatus::Ok
            );
            assert_eq!(labels[0], labels[1]);
            assert_ne!(labels[0], labels[3]);
            assert!(labels.iter().all(|&l| l >= 0));
            tpml_dbscan_free(h);
        }
    }

    #[test]
    fn ridge_predicts_a_linear_relation() {
        let x = [0.0_f64, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0, 2.0, 3.0, 1.0];
        let y: Vec<f64> = (0..5).map(|i| 2.0 * x[i * 2] - x[i * 2 + 1]).collect();
        unsafe {
            let h = tpml_ridge_new(1e-9);
            assert!(!h.is_null());
            assert_eq!(
                tpml_ridge_fit(h, x.as_ptr(), 5, 2, y.as_ptr()),
                TpmlStatus::Ok
            );
            let mut coef = [0.0_f64; 2];
            assert_eq!(tpml_ridge_coef(h, coef.as_mut_ptr(), 2), TpmlStatus::Ok);
            assert!((coef[0] - 2.0).abs() < 1e-4 && (coef[1] + 1.0).abs() < 1e-4);
            let mut pred = [0.0_f64; 5];
            assert_eq!(
                tpml_ridge_predict(h, x.as_ptr(), 5, 2, pred.as_mut_ptr(), 5),
                TpmlStatus::Ok
            );
            assert!((pred[0] - y[0]).abs() < 1e-4);
            tpml_ridge_free(h);
        }
    }

    #[test]
    fn unfitted_and_bad_lengths_are_reported() {
        let x = [0.0_f64, 1.0];
        unsafe {
            let h = tpml_ridge_new(0.1);
            let mut out = [0.0_f64; 1];
            assert_eq!(
                tpml_ridge_predict(h, x.as_ptr(), 1, 2, out.as_mut_ptr(), 1),
                TpmlStatus::NotFitted
            );
            // A wrong output length is reported, not written.
            assert_eq!(
                tpml_ridge_fit(h, x.as_ptr(), 1, 2, [1.0_f64].as_ptr()),
                TpmlStatus::Ok
            );
            assert_eq!(
                tpml_ridge_predict(h, x.as_ptr(), 1, 2, out.as_mut_ptr(), 3),
                TpmlStatus::InvalidArgument
            );
            tpml_ridge_free(h);
        }
        assert!(tpml_stump_forest_new(0, 0).is_null());
        assert!(tpml_elastic_net_new(0.1, 1.5, 10, 1e-6).is_null());
    }

    #[test]
    fn lasso_zeroes_an_unused_feature() {
        let x = [
            0.0_f64, 5.0, 1.0, -3.0, 2.0, 7.0, 3.0, 1.0, 4.0, -2.0, 5.0, 0.5,
        ];
        let y: Vec<f64> = (0..6).map(|i| 2.0 * x[i * 2]).collect();
        unsafe {
            let h = tpml_lasso_new(0.5, 10000, 1e-10);
            assert!(!h.is_null());
            assert_eq!(
                tpml_lasso_fit(h, x.as_ptr(), 6, 2, y.as_ptr()),
                TpmlStatus::Ok
            );
            let mut coef = [0.0_f64; 2];
            assert_eq!(tpml_lasso_coef(h, coef.as_mut_ptr(), 2), TpmlStatus::Ok);
            assert!(coef[1].abs() < 1e-12);
            tpml_lasso_free(h);
        }
        assert!(tpml_lasso_new(-1.0, 10, 1e-6).is_null());
    }
}
