// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! C ABI for KMeans and PCA, used by the Go binding and any C-compatible caller. The Phase 8
//! estimators live in the `models` module, with the same conventions.
//!
//! Conventions:
//!
//! * Data is `f64`, row-major, passed as a pointer plus `rows` and `cols`.
//! * Every function returns a [`TpmlStatus`]. `Ok` is 0. On failure, call
//!   `tpml_last_error_message` for a description. The message is valid until the next
//!   call on the same thread.
//! * Output buffers are allocated by the caller. Each getter takes the element count
//!   and returns an error if it does not match.
//! * Objects are opaque pointers from `*_new`, released with `*_free`.
//! * No Rust panic crosses the boundary. A panic becomes `Internal`.

use std::cell::RefCell;
use std::ffi::{c_char, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

use tpt_ml_proven_clustering::{Init, KMeans};
use tpt_ml_proven_core::error::Error;
use tpt_ml_proven_core::{DenseView, Predictor, Transformer, UnsupervisedEstimator};
use tpt_ml_proven_decomposition::Pca;

mod models;

/// Result of a call. `Ok` is zero.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpmlStatus {
    /// The call succeeded.
    Ok = 0,
    /// A required pointer was null.
    NullPointer = 1,
    /// A parameter, shape, or buffer length was invalid.
    InvalidArgument = 2,
    /// The estimator has not been fitted.
    NotFitted = 3,
    /// The input contained NaN or infinity.
    NonFinite = 4,
    /// An internal error, including a caught panic.
    Internal = 5,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(msg: &str) {
    let safe = msg.replace('\0', " ");
    let c = CString::new(safe).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = c);
}

/// Message for the most recent failure on this thread, as a NUL-terminated string.
/// The pointer is valid until the next call on the same thread.
#[no_mangle]
pub extern "C" fn tpml_last_error_message() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

struct Failure {
    status: TpmlStatus,
    message: String,
}

impl Failure {
    fn new(status: TpmlStatus, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

impl From<Error> for Failure {
    fn from(e: Error) -> Self {
        let status = match e {
            Error::NotFitted => TpmlStatus::NotFitted,
            Error::NonFinite { .. } => TpmlStatus::NonFinite,
            Error::Contract(_) => TpmlStatus::Internal,
            _ => TpmlStatus::InvalidArgument,
        };
        Failure::new(status, e.to_string())
    }
}

impl From<tpt_ml_proven_core::ShapeError> for Failure {
    fn from(e: tpt_ml_proven_core::ShapeError) -> Self {
        Failure::new(TpmlStatus::InvalidArgument, e.to_string())
    }
}

/// Runs `f`, records any failure, and converts panics to `Internal`.
fn run<F>(f: F) -> TpmlStatus
where
    F: FnOnce() -> Result<(), Failure>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => {
            set_error("");
            TpmlStatus::Ok
        }
        Ok(Err(fail)) => {
            set_error(&fail.message);
            fail.status
        }
        Err(_) => {
            set_error("internal panic in tpt-ml-proven");
            TpmlStatus::Internal
        }
    }
}

/// Borrows `rows * cols` values from a caller pointer.
///
/// # Safety
/// `data` must be valid for reads of `rows * cols` `f64` values.
unsafe fn input<'a>(data: *const f64, rows: usize, cols: usize) -> Result<&'a [f64], Failure> {
    if data.is_null() {
        return Err(Failure::new(
            TpmlStatus::NullPointer,
            "data pointer is null",
        ));
    }
    let n = rows
        .checked_mul(cols)
        .ok_or_else(|| Failure::new(TpmlStatus::InvalidArgument, "rows * cols overflows"))?;
    Ok(std::slice::from_raw_parts(data, n))
}

/// Writes `values` into a caller buffer of exactly `len` elements.
///
/// # Safety
/// `out` must be valid for writes of `len` `T` values.
unsafe fn output<T: Copy>(out: *mut T, len: usize, values: &[T]) -> Result<(), Failure> {
    if out.is_null() {
        return Err(Failure::new(
            TpmlStatus::NullPointer,
            "output pointer is null",
        ));
    }
    if len != values.len() {
        return Err(Failure::new(
            TpmlStatus::InvalidArgument,
            format!("output length {len} does not match {}", values.len()),
        ));
    }
    std::ptr::copy_nonoverlapping(values.as_ptr(), out, len);
    Ok(())
}

// ---------------------------------------------------------------- KMeans

/// Opaque KMeans handle.
pub struct TpmlKMeans {
    n_clusters: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    threads: usize,
    fitted: Option<KMeans<f64>>,
}

/// Creates a KMeans estimator (k-means++ initialisation). Free it with
/// `tpml_kmeans_free`. Returns null on invalid input.
#[no_mangle]
pub extern "C" fn tpml_kmeans_new(
    n_clusters: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    threads: usize,
) -> *mut TpmlKMeans {
    if n_clusters == 0 || max_iter == 0 || !tol.is_finite() || tol < 0.0 || threads == 0 {
        set_error("invalid KMeans parameters");
        return std::ptr::null_mut();
    }
    set_error("");
    Box::into_raw(Box::new(TpmlKMeans {
        n_clusters,
        max_iter,
        tol,
        seed,
        threads,
        fitted: None,
    }))
}

/// Releases a handle from `tpml_kmeans_new`. Null is ignored.
///
/// # Safety
/// `h` must be null or a handle from `tpml_kmeans_new` that has not been freed.
#[no_mangle]
pub unsafe extern "C" fn tpml_kmeans_free(h: *mut TpmlKMeans) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Fits to `rows x cols` row-major data.
///
/// # Safety
/// `h` must be a live handle. `data` must be valid for `rows * cols` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_kmeans_fit(
    h: *mut TpmlKMeans,
    data: *const f64,
    rows: usize,
    cols: usize,
) -> TpmlStatus {
    run(|| {
        let h = h
            .as_mut()
            .ok_or_else(|| Failure::new(TpmlStatus::NullPointer, "handle is null"))?;
        let slice = input(data, rows, cols)?;
        let mut km = KMeans::<f64>::new(h.n_clusters)
            .with_max_iter(h.max_iter)
            .with_tol(h.tol)
            .with_init(Init::KMeansPlusPlus)
            .with_seed(h.seed)
            .with_threads(h.threads);
        UnsupervisedEstimator::fit(&mut km, DenseView::new(slice, rows, cols)?)?;
        h.fitted = Some(km);
        Ok(())
    })
}

/// Nearest cluster for each row. `out` must hold `rows` values.
///
/// # Safety
/// `h` must be a live handle. `data` is valid for `rows * cols` reads and `out` for
/// `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_kmeans_predict(
    h: *const TpmlKMeans,
    data: *const f64,
    rows: usize,
    cols: usize,
    out: *mut usize,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h
            .as_ref()
            .ok_or_else(|| Failure::new(TpmlStatus::NullPointer, "handle is null"))?;
        let km = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        let slice = input(data, rows, cols)?;
        let labels = km.predict(DenseView::new(slice, rows, cols)?)?;
        output(out, out_len, &labels)
    })
}

/// Squared distances to the centres, `rows x n_clusters` into `out`.
///
/// # Safety
/// As `tpml_kmeans_predict`, with `out` valid for `out_len` `f64` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_kmeans_transform(
    h: *const TpmlKMeans,
    data: *const f64,
    rows: usize,
    cols: usize,
    out: *mut f64,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h
            .as_ref()
            .ok_or_else(|| Failure::new(TpmlStatus::NullPointer, "handle is null"))?;
        let km = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        let slice = input(data, rows, cols)?;
        let d = km.transform(DenseView::new(slice, rows, cols)?)?;
        output(out, out_len, &d.into_vec())
    })
}

/// Training labels. `out` must hold as many values as there were training rows.
///
/// # Safety
/// `h` must be a live handle and `out` valid for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_kmeans_labels(
    h: *const TpmlKMeans,
    out: *mut usize,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h
            .as_ref()
            .ok_or_else(|| Failure::new(TpmlStatus::NullPointer, "handle is null"))?;
        let km = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        output(out, out_len, km.labels().unwrap_or(&[]))
    })
}

/// Fitted centres, `n_clusters x cols` row-major into `out`.
///
/// # Safety
/// `h` must be a live handle and `out` valid for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_kmeans_centers(
    h: *const TpmlKMeans,
    out: *mut f64,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h
            .as_ref()
            .ok_or_else(|| Failure::new(TpmlStatus::NullPointer, "handle is null"))?;
        let km = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        let c = km.cluster_centers().ok_or(Error::NotFitted)?;
        output(out, out_len, &c.clone().into_vec())
    })
}

/// Sum of squared distances to the assigned centres.
///
/// # Safety
/// `h` must be a live handle and `out` must be valid for one write.
#[no_mangle]
pub unsafe extern "C" fn tpml_kmeans_inertia(h: *const TpmlKMeans, out: *mut f64) -> TpmlStatus {
    run(|| {
        let h = h
            .as_ref()
            .ok_or_else(|| Failure::new(TpmlStatus::NullPointer, "handle is null"))?;
        let km = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        output(out, 1, &[km.inertia().ok_or(Error::NotFitted)?])
    })
}

// ---------------------------------------------------------------- PCA

/// Opaque PCA handle.
pub struct TpmlPca {
    n_components: usize,
    fitted: Option<Pca<f64>>,
}

/// Creates a PCA keeping `n_components` components. Free with `tpml_pca_free`.
#[no_mangle]
pub extern "C" fn tpml_pca_new(n_components: usize) -> *mut TpmlPca {
    if n_components == 0 {
        set_error("n_components must be at least 1");
        return std::ptr::null_mut();
    }
    set_error("");
    Box::into_raw(Box::new(TpmlPca {
        n_components,
        fitted: None,
    }))
}

/// Releases a handle from `tpml_pca_new`. Null is ignored.
///
/// # Safety
/// `h` must be null or a handle from `tpml_pca_new` that has not been freed.
#[no_mangle]
pub unsafe extern "C" fn tpml_pca_free(h: *mut TpmlPca) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Fits to `rows x cols` row-major data.
///
/// # Safety
/// `h` must be a live handle. `data` must be valid for `rows * cols` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_pca_fit(
    h: *mut TpmlPca,
    data: *const f64,
    rows: usize,
    cols: usize,
) -> TpmlStatus {
    run(|| {
        let h = h
            .as_mut()
            .ok_or_else(|| Failure::new(TpmlStatus::NullPointer, "handle is null"))?;
        let slice = input(data, rows, cols)?;
        let mut p = Pca::<f64>::new(h.n_components);
        UnsupervisedEstimator::fit(&mut p, DenseView::new(slice, rows, cols)?)?;
        h.fitted = Some(p);
        Ok(())
    })
}

/// Projection onto the kept components, `rows x n_components` into `out`.
///
/// # Safety
/// `h` must be a live handle. `data` is valid for `rows * cols` reads and `out` for
/// `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_pca_transform(
    h: *const TpmlPca,
    data: *const f64,
    rows: usize,
    cols: usize,
    out: *mut f64,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h
            .as_ref()
            .ok_or_else(|| Failure::new(TpmlStatus::NullPointer, "handle is null"))?;
        let p = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        let slice = input(data, rows, cols)?;
        let z = p.transform(DenseView::new(slice, rows, cols)?)?;
        output(out, out_len, &z.into_vec())
    })
}

/// Principal axes, `n_components x cols` row-major into `out`.
///
/// # Safety
/// `h` must be a live handle and `out` valid for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_pca_components(
    h: *const TpmlPca,
    out: *mut f64,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h
            .as_ref()
            .ok_or_else(|| Failure::new(TpmlStatus::NullPointer, "handle is null"))?;
        let p = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        let c = p.components().ok_or(Error::NotFitted)?;
        output(out, out_len, &c.clone().into_vec())
    })
}

/// Variance along each kept component. `out` holds `n_components` values.
///
/// # Safety
/// `h` must be a live handle and `out` valid for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_pca_explained_variance(
    h: *const TpmlPca,
    out: *mut f64,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h
            .as_ref()
            .ok_or_else(|| Failure::new(TpmlStatus::NullPointer, "handle is null"))?;
        let p = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        output(
            out,
            out_len,
            p.explained_variance().ok_or(Error::NotFitted)?,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_round_trip_through_the_c_abi() {
        let data = [0.0_f64, 0.0, 0.1, 0.0, 10.0, 10.0, 10.1, 10.0];
        unsafe {
            let h = tpml_kmeans_new(2, 300, 1e-4, 1, 1);
            assert!(!h.is_null());
            assert_eq!(tpml_kmeans_fit(h, data.as_ptr(), 4, 2), TpmlStatus::Ok);
            let mut labels = [0_usize; 4];
            assert_eq!(
                tpml_kmeans_labels(h, labels.as_mut_ptr(), 4),
                TpmlStatus::Ok
            );
            assert_eq!(labels[0], labels[1]);
            assert_ne!(labels[0], labels[2]);
            let mut centers = [0.0_f64; 4];
            assert_eq!(
                tpml_kmeans_centers(h, centers.as_mut_ptr(), 4),
                TpmlStatus::Ok
            );
            tpml_kmeans_free(h);
        }
    }

    #[test]
    fn wrong_buffer_length_is_an_invalid_argument_with_a_message() {
        let data = [0.0_f64, 1.0, 2.0, 3.0];
        unsafe {
            let h = tpml_kmeans_new(2, 300, 1e-4, 1, 1);
            assert_eq!(tpml_kmeans_fit(h, data.as_ptr(), 2, 2), TpmlStatus::Ok);
            let mut labels = [0_usize; 5];
            let status = tpml_kmeans_labels(h, labels.as_mut_ptr(), 5);
            assert_eq!(status, TpmlStatus::InvalidArgument);
            let msg = std::ffi::CStr::from_ptr(tpml_last_error_message()).to_string_lossy();
            assert!(msg.contains("output length 5"), "{msg}");
            tpml_kmeans_free(h);
        }
    }

    #[test]
    fn calls_before_fit_report_not_fitted() {
        unsafe {
            let h = tpml_kmeans_new(2, 300, 1e-4, 1, 1);
            let mut labels = [0_usize; 2];
            assert_eq!(
                tpml_kmeans_labels(h, labels.as_mut_ptr(), 2),
                TpmlStatus::NotFitted
            );
            tpml_kmeans_free(h);
        }
    }

    #[test]
    fn null_pointers_and_bad_parameters_are_rejected() {
        unsafe {
            assert!(tpml_kmeans_new(0, 300, 1e-4, 1, 1).is_null());
            assert!(tpml_kmeans_new(2, 300, f64::NAN, 1, 1).is_null());
            assert_eq!(
                tpml_kmeans_fit(std::ptr::null_mut(), std::ptr::null(), 1, 1),
                TpmlStatus::NullPointer
            );
            let h = tpml_kmeans_new(2, 300, 1e-4, 1, 1);
            assert_eq!(
                tpml_kmeans_fit(h, std::ptr::null(), 2, 2),
                TpmlStatus::NullPointer
            );
            tpml_kmeans_free(h);
        }
    }

    #[test]
    fn non_finite_input_is_reported() {
        let data = [0.0_f64, f64::NAN, 1.0, 2.0];
        unsafe {
            let h = tpml_kmeans_new(1, 300, 1e-4, 1, 1);
            assert_eq!(
                tpml_kmeans_fit(h, data.as_ptr(), 2, 2),
                TpmlStatus::NonFinite
            );
            tpml_kmeans_free(h);
        }
    }

    #[test]
    fn pca_round_trip_through_the_c_abi() {
        let data = [1.0_f64, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0];
        unsafe {
            let h = tpml_pca_new(1);
            assert_eq!(tpml_pca_fit(h, data.as_ptr(), 4, 2), TpmlStatus::Ok);
            let mut z = [0.0_f64; 4];
            assert_eq!(
                tpml_pca_transform(h, data.as_ptr(), 4, 2, z.as_mut_ptr(), 4),
                TpmlStatus::Ok
            );
            let mut ev = [0.0_f64; 1];
            assert_eq!(
                tpml_pca_explained_variance(h, ev.as_mut_ptr(), 1),
                TpmlStatus::Ok
            );
            assert!(ev[0] > 0.0);
            tpml_pca_free(h);
        }
    }
}
