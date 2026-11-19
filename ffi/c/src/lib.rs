// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! C ABI for KMeans and PCA, used by the Go binding and any C-compatible caller. The Phase 8
//! estimators live in the `models` module, with the same conventions.
//!
//! Conventions:
//!
//! * Data is row-major, passed as a pointer plus `rows` and `cols`. It is `f64`, except for the
//!   `_fit_f32` functions, which take `f32` and make a single-precision model. Predictions and
//!   transforms always take `f64`, which a single-precision model narrows, and return `f64`.
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
use tpt_ml_proven_core::{DenseView, Float, Predictor, Transformer, UnsupervisedEstimator};
use tpt_ml_proven_decomposition::Pca;

mod models;
mod preprocessing;
mod trees;

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

/// The library version as a NUL-terminated string, for example "0.0.0". The pointer is
/// static and never freed.
#[no_mangle]
pub extern "C" fn tpml_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
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
unsafe fn input<'a, T>(data: *const T, rows: usize, cols: usize) -> Result<&'a [T], Failure> {
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

/// The fitted KMeans, in the precision of the last fit.
enum KState {
    F64(KMeans<f64>),
    F32(KMeans<f32>),
}

/// Opaque KMeans handle.
pub struct TpmlKMeans {
    n_clusters: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    threads: usize,
    fitted: Option<KState>,
}

impl TpmlKMeans {
    fn estimator<F: Float>(&self) -> KMeans<F> {
        KMeans::<F>::new(self.n_clusters)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_init(Init::KMeansPlusPlus)
            .with_seed(self.seed)
            .with_threads(self.threads)
    }
}

/// Narrows `f64` input for a single-precision model.
fn narrow(values: &[f64]) -> Vec<f32> {
    values.iter().map(|&v| v as f32).collect()
}

/// Widens single-precision results to `f64` for the caller.
fn widen(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(v)).collect()
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

/// Fits to `rows x cols` row-major `f64` data. The model is double precision.
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
        let mut km = h.estimator::<f64>();
        UnsupervisedEstimator::fit(&mut km, DenseView::new(slice, rows, cols)?)?;
        h.fitted = Some(KState::F64(km));
        Ok(())
    })
}

/// Fits to `rows x cols` row-major `f32` data. The model is single precision, which halves the
/// memory the data needs. Predictions still take `f64` input, which is narrowed.
///
/// # Safety
/// `h` must be a live handle. `data` must be valid for `rows * cols` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_kmeans_fit_f32(
    h: *mut TpmlKMeans,
    data: *const f32,
    rows: usize,
    cols: usize,
) -> TpmlStatus {
    run(|| {
        let h = h
            .as_mut()
            .ok_or_else(|| Failure::new(TpmlStatus::NullPointer, "handle is null"))?;
        let slice = input(data, rows, cols)?;
        let mut km = h.estimator::<f32>();
        UnsupervisedEstimator::fit(&mut km, DenseView::new(slice, rows, cols)?)?;
        h.fitted = Some(KState::F32(km));
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
        let slice = input(data, rows, cols)?;
        let labels = match h.fitted.as_ref().ok_or(Error::NotFitted)? {
            KState::F64(km) => km.predict(DenseView::new(slice, rows, cols)?)?,
            KState::F32(km) => {
                let narrowed = narrow(slice);
                km.predict(DenseView::new(narrowed.as_slice(), rows, cols)?)?
            }
        };
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
        let slice = input(data, rows, cols)?;
        let values: Vec<f64> = match h.fitted.as_ref().ok_or(Error::NotFitted)? {
            KState::F64(km) => km.transform(DenseView::new(slice, rows, cols)?)?.into_vec(),
            KState::F32(km) => {
                let narrowed = narrow(slice);
                let d = km.transform(DenseView::new(narrowed.as_slice(), rows, cols)?)?;
                widen(&d.into_vec())
            }
        };
        output(out, out_len, &values)
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
        let labels: &[usize] = match h.fitted.as_ref().ok_or(Error::NotFitted)? {
            KState::F64(km) => km.labels().unwrap_or(&[]),
            KState::F32(km) => km.labels().unwrap_or(&[]),
        };
        output(out, out_len, labels)
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
        let values: Vec<f64> = match h.fitted.as_ref().ok_or(Error::NotFitted)? {
            KState::F64(km) => km
                .cluster_centers()
                .ok_or(Error::NotFitted)?
                .clone()
                .into_vec(),
            KState::F32(km) => {
                let c = km.cluster_centers().ok_or(Error::NotFitted)?.clone();
                widen(&c.into_vec())
            }
        };
        output(out, out_len, &values)
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
        let inertia = match h.fitted.as_ref().ok_or(Error::NotFitted)? {
            KState::F64(km) => km.inertia().ok_or(Error::NotFitted)?,
            KState::F32(km) => f64::from(km.inertia().ok_or(Error::NotFitted)?),
        };
        output(out, 1, &[inertia])
    })
}

/// Writes the number of Lloyd iterations the last fit used into `out`.
///
/// # Safety
/// `h` must be a live handle and `out` must be valid for one write.
#[no_mangle]
pub unsafe extern "C" fn tpml_kmeans_n_iter(h: *const TpmlKMeans, out: *mut usize) -> TpmlStatus {
    run(|| {
        let h = h
            .as_ref()
            .ok_or_else(|| Failure::new(TpmlStatus::NullPointer, "handle is null"))?;
        let n = match h.fitted.as_ref().ok_or(Error::NotFitted)? {
            KState::F64(km) => km.n_iter(),
            KState::F32(km) => km.n_iter(),
        };
        output(out, 1, &[n.ok_or(Error::NotFitted)?])
    })
}

// ---------------------------------------------------------------- PCA

/// The fitted PCA, in the precision of the last fit.
enum PState {
    F64(Pca<f64>),
    F32(Pca<f32>),
}

/// Opaque PCA handle.
pub struct TpmlPca {
    n_components: usize,
    fitted: Option<PState>,
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

/// Fits to `rows x cols` row-major `f64` data. The model is double precision.
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
        h.fitted = Some(PState::F64(p));
        Ok(())
    })
}

/// Fits to `rows x cols` row-major `f32` data. The model is single precision. Transforms still
/// take `f64` input, which is narrowed.
///
/// # Safety
/// `h` must be a live handle. `data` must be valid for `rows * cols` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_pca_fit_f32(
    h: *mut TpmlPca,
    data: *const f32,
    rows: usize,
    cols: usize,
) -> TpmlStatus {
    run(|| {
        let h = h
            .as_mut()
            .ok_or_else(|| Failure::new(TpmlStatus::NullPointer, "handle is null"))?;
        let slice = input(data, rows, cols)?;
        let mut p = Pca::<f32>::new(h.n_components);
        UnsupervisedEstimator::fit(&mut p, DenseView::new(slice, rows, cols)?)?;
        h.fitted = Some(PState::F32(p));
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
        let slice = input(data, rows, cols)?;
        let values: Vec<f64> = match h.fitted.as_ref().ok_or(Error::NotFitted)? {
            PState::F64(p) => p.transform(DenseView::new(slice, rows, cols)?)?.into_vec(),
            PState::F32(p) => {
                let narrowed = narrow(slice);
                let z = p.transform(DenseView::new(narrowed.as_slice(), rows, cols)?)?;
                widen(&z.into_vec())
            }
        };
        output(out, out_len, &values)
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
        let values: Vec<f64> = match h.fitted.as_ref().ok_or(Error::NotFitted)? {
            PState::F64(p) => p.components().ok_or(Error::NotFitted)?.clone().into_vec(),
            PState::F32(p) => {
                let c = p.components().ok_or(Error::NotFitted)?.clone();
                widen(&c.into_vec())
            }
        };
        output(out, out_len, &values)
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
        let values: Vec<f64> = match h.fitted.as_ref().ok_or(Error::NotFitted)? {
            PState::F64(p) => p.explained_variance().ok_or(Error::NotFitted)?.to_vec(),
            PState::F32(p) => widen(p.explained_variance().ok_or(Error::NotFitted)?),
        };
        output(out, out_len, &values)
    })
}

/// Maps `rows x n_components` scores back to the feature space, `rows x cols` into `out`.
/// `cols` must equal `n_components`, and `out_len` must hold `rows` times the feature count.
///
/// # Safety
/// `h` must be a live handle, `z` valid for `rows * cols` reads, and `out` valid for `out_len`
/// writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_pca_inverse_transform(
    h: *const TpmlPca,
    z: *const f64,
    rows: usize,
    cols: usize,
    out: *mut f64,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h
            .as_ref()
            .ok_or_else(|| Failure::new(TpmlStatus::NullPointer, "handle is null"))?;
        let slice = input(z, rows, cols)?;
        let values: Vec<f64> = match h.fitted.as_ref().ok_or(Error::NotFitted)? {
            PState::F64(p) => p
                .inverse_transform(&DenseView::new(slice, rows, cols)?)?
                .into_vec(),
            PState::F32(p) => {
                let narrowed = narrow(slice);
                let x = p.inverse_transform(&DenseView::new(narrowed.as_slice(), rows, cols)?)?;
                widen(&x.into_vec())
            }
        };
        output(out, out_len, &values)
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

    #[test]
    fn pca_inverse_transform_recovers_rank_one_data() {
        // The second column is twice the first, so one component reconstructs every row.
        let data = [1.0_f64, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0];
        unsafe {
            let h = tpml_pca_new(1);
            assert_eq!(tpml_pca_fit(h, data.as_ptr(), 4, 2), TpmlStatus::Ok);
            let mut z = [0.0_f64; 4];
            assert_eq!(
                tpml_pca_transform(h, data.as_ptr(), 4, 2, z.as_mut_ptr(), 4),
                TpmlStatus::Ok
            );
            let mut back = [0.0_f64; 8];
            assert_eq!(
                tpml_pca_inverse_transform(h, z.as_ptr(), 4, 1, back.as_mut_ptr(), 8),
                TpmlStatus::Ok
            );
            for (a, b) in back.iter().zip(data.iter()) {
                assert!((a - b).abs() < 1e-9, "{a} != {b}");
            }
            tpml_pca_free(h);
        }
    }

    #[test]
    fn kmeans_n_iter_is_reported_after_fit() {
        let data = [0.0_f64, 0.0, 0.1, 0.0, 10.0, 10.0, 10.1, 10.0];
        unsafe {
            let h = tpml_kmeans_new(2, 300, 1e-4, 1, 1);
            let mut n = 0_usize;
            assert_eq!(tpml_kmeans_n_iter(h, &mut n), TpmlStatus::NotFitted);
            assert_eq!(tpml_kmeans_fit(h, data.as_ptr(), 4, 2), TpmlStatus::Ok);
            assert_eq!(tpml_kmeans_n_iter(h, &mut n), TpmlStatus::Ok);
            assert!(n >= 1, "n_iter was {n}");
            tpml_kmeans_free(h);
        }
    }

    #[test]
    fn kmeans_f32_fit_separates_the_blobs() {
        let data = [0.0_f32, 0.0, 0.1, 0.0, 10.0, 10.0, 10.1, 10.0];
        let query = [0.0_f64, 0.0, 10.0, 10.0];
        unsafe {
            let h = tpml_kmeans_new(2, 300, 1e-4, 1, 1);
            assert_eq!(tpml_kmeans_fit_f32(h, data.as_ptr(), 4, 2), TpmlStatus::Ok);
            let mut labels = [0_usize; 4];
            assert_eq!(
                tpml_kmeans_labels(h, labels.as_mut_ptr(), 4),
                TpmlStatus::Ok
            );
            assert_eq!(labels[0], labels[1]);
            assert_eq!(labels[2], labels[3]);
            assert_ne!(labels[0], labels[2]);
            // Predictions take f64 input, which the single-precision model narrows.
            let mut predicted = [0_usize; 2];
            assert_eq!(
                tpml_kmeans_predict(h, query.as_ptr(), 2, 2, predicted.as_mut_ptr(), 2),
                TpmlStatus::Ok
            );
            assert_eq!(predicted, [labels[0], labels[2]]);
            let mut inertia = 0.0_f64;
            assert_eq!(tpml_kmeans_inertia(h, &mut inertia), TpmlStatus::Ok);
            assert!(inertia.is_finite() && inertia >= 0.0);
            tpml_kmeans_free(h);
        }
    }

    #[test]
    fn pca_f32_matches_f64_within_single_precision() {
        let data64 = [1.0_f64, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0];
        let data32: Vec<f32> = data64.iter().map(|&v| v as f32).collect();
        unsafe {
            let h64 = tpml_pca_new(1);
            assert_eq!(tpml_pca_fit(h64, data64.as_ptr(), 4, 2), TpmlStatus::Ok);
            let h32 = tpml_pca_new(1);
            assert_eq!(tpml_pca_fit_f32(h32, data32.as_ptr(), 4, 2), TpmlStatus::Ok);

            let mut z64 = [0.0_f64; 4];
            let mut z32 = [0.0_f64; 4];
            assert_eq!(
                tpml_pca_transform(h64, data64.as_ptr(), 4, 2, z64.as_mut_ptr(), 4),
                TpmlStatus::Ok
            );
            assert_eq!(
                tpml_pca_transform(h32, data64.as_ptr(), 4, 2, z32.as_mut_ptr(), 4),
                TpmlStatus::Ok
            );
            for (a, b) in z64.iter().zip(z32.iter()) {
                assert!((a.abs() - b.abs()).abs() < 1e-4, "{a} vs {b}");
            }

            let mut back = [0.0_f64; 8];
            assert_eq!(
                tpml_pca_inverse_transform(h32, z32.as_ptr(), 4, 1, back.as_mut_ptr(), 8),
                TpmlStatus::Ok
            );
            for (a, b) in back.iter().zip(data64.iter()) {
                assert!((a - b).abs() < 1e-4, "{a} != {b}");
            }
            tpml_pca_free(h64);
            tpml_pca_free(h32);
        }
    }
}
