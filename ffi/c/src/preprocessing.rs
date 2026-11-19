// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! C ABI for the preprocessing transformers: the scalers, the encoders and the imputer.
//!
//! Each transformer has `new`, `fit`, `transform`, `out_cols` (the number of values per output
//! row, which the caller needs to size the output buffer) and `free`.

use std::ffi::c_double;

use tpt_ml_proven_core::error::Error;
use tpt_ml_proven_core::Dense;
use tpt_ml_proven_core::{DenseView, Transformer, UnsupervisedEstimator};
use tpt_ml_proven_preprocessing::{
    ImputeStrategy, MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, SimpleImputer,
    StandardScaler,
};

use super::models::null_handle;
use super::{input, output, run, Failure, TpmlStatus};

/// How `SimpleImputer` chooses its fill value. C callers construct these; Rust only matches them.
#[repr(C)]
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpmlImputeStrategy {
    /// The mean of the observed values.
    Mean = 0,
    /// The median of the observed values.
    Median = 1,
    /// The most frequent observed value.
    MostFrequent = 2,
    /// A fixed value, given to `tpml_simple_imputer_new`.
    Constant = 3,
}

/// Writes the transformed matrix of a fitted transformer into `out`.
fn transform_into<T: Transformer<Float = f64, Output = Dense<f64>>>(
    model: &T,
    data: *const c_double,
    rows: usize,
    cols: usize,
    out: *mut c_double,
    out_len: usize,
) -> Result<(), Failure> {
    // SAFETY: the caller of the exported function guarantees `data` is valid for `rows * cols`.
    let x = unsafe { input(data, rows, cols)? };
    let z = Transformer::transform(model, DenseView::new(x, rows, cols)?)?;
    // SAFETY: the caller guarantees `out` is valid for `out_len` writes.
    unsafe { output(out, out_len, z.as_slice()) }
}

// ---------------------------------------------------------------- StandardScaler

/// Opaque standard scaler handle.
pub struct TpmlStandardScaler {
    fitted: Option<StandardScaler<f64>>,
}

/// Creates a standard scaler. Free it with `tpml_standard_scaler_free`.
#[no_mangle]
pub extern "C" fn tpml_standard_scaler_new() -> *mut TpmlStandardScaler {
    Box::into_raw(Box::new(TpmlStandardScaler { fitted: None }))
}

/// Frees a standard scaler handle. Null is ignored.
///
/// # Safety
/// `h` must be null or a live handle from `tpml_standard_scaler_new`.
#[no_mangle]
pub unsafe extern "C" fn tpml_standard_scaler_free(h: *mut TpmlStandardScaler) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Learns the per-feature mean and scale from a row-major `rows × cols` matrix.
///
/// # Safety
/// `data` is valid for `rows * cols` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_standard_scaler_fit(
    h: *mut TpmlStandardScaler,
    data: *const c_double,
    rows: usize,
    cols: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        let x = input(data, rows, cols)?;
        let mut m = StandardScaler::<f64>::new();
        UnsupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols)?)?;
        h.fitted = Some(m);
        Ok(())
    })
}

/// Transforms a row-major `rows × cols` matrix. `out` receives `rows × out_cols` values.
///
/// # Safety
/// `data` is valid for `rows * cols` reads and `out` for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_standard_scaler_transform(
    h: *const TpmlStandardScaler,
    data: *const c_double,
    rows: usize,
    cols: usize,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        transform_into(m, data, rows, cols, out, out_len)
    })
}

/// Writes the number of values per output row into `out`.
///
/// # Safety
/// `h` is a live handle and `out` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn tpml_standard_scaler_out_cols(
    h: *const TpmlStandardScaler,
    out: *mut usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        output(out, 1, &[m.n_features().unwrap_or(0)])
    })
}

// ---------------------------------------------------------------- MinMaxScaler

/// Opaque min-max scaler handle.
pub struct TpmlMinMaxScaler {
    fitted: Option<MinMaxScaler<f64>>,
}

/// Creates a min-max scaler. Free it with `tpml_min_max_scaler_free`.
#[no_mangle]
pub extern "C" fn tpml_min_max_scaler_new() -> *mut TpmlMinMaxScaler {
    Box::into_raw(Box::new(TpmlMinMaxScaler { fitted: None }))
}

/// Frees a min-max scaler handle. Null is ignored.
///
/// # Safety
/// `h` must be null or a live handle from `tpml_min_max_scaler_new`.
#[no_mangle]
pub unsafe extern "C" fn tpml_min_max_scaler_free(h: *mut TpmlMinMaxScaler) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Learns the per-feature range from a row-major `rows × cols` matrix.
///
/// # Safety
/// `data` is valid for `rows * cols` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_min_max_scaler_fit(
    h: *mut TpmlMinMaxScaler,
    data: *const c_double,
    rows: usize,
    cols: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        let x = input(data, rows, cols)?;
        let mut m = MinMaxScaler::<f64>::new();
        UnsupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols)?)?;
        h.fitted = Some(m);
        Ok(())
    })
}

/// Transforms a row-major `rows × cols` matrix. `out` receives `rows × out_cols` values.
///
/// # Safety
/// `data` is valid for `rows * cols` reads and `out` for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_min_max_scaler_transform(
    h: *const TpmlMinMaxScaler,
    data: *const c_double,
    rows: usize,
    cols: usize,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        transform_into(m, data, rows, cols, out, out_len)
    })
}

/// Writes the number of values per output row into `out`.
///
/// # Safety
/// `h` is a live handle and `out` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn tpml_min_max_scaler_out_cols(
    h: *const TpmlMinMaxScaler,
    out: *mut usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        output(out, 1, &[m.n_features().unwrap_or(0)])
    })
}

// ---------------------------------------------------------------- RobustScaler

/// Opaque robust scaler handle.
pub struct TpmlRobustScaler {
    fitted: Option<RobustScaler<f64>>,
}

/// Creates a robust scaler. Free it with `tpml_robust_scaler_free`.
#[no_mangle]
pub extern "C" fn tpml_robust_scaler_new() -> *mut TpmlRobustScaler {
    Box::into_raw(Box::new(TpmlRobustScaler { fitted: None }))
}

/// Frees a robust scaler handle. Null is ignored.
///
/// # Safety
/// `h` must be null or a live handle from `tpml_robust_scaler_new`.
#[no_mangle]
pub unsafe extern "C" fn tpml_robust_scaler_free(h: *mut TpmlRobustScaler) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Learns the per-feature median and interquartile range from a row-major `rows × cols` matrix.
///
/// # Safety
/// `data` is valid for `rows * cols` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_robust_scaler_fit(
    h: *mut TpmlRobustScaler,
    data: *const c_double,
    rows: usize,
    cols: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        let x = input(data, rows, cols)?;
        let mut m = RobustScaler::<f64>::new();
        UnsupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols)?)?;
        h.fitted = Some(m);
        Ok(())
    })
}

/// Transforms a row-major `rows × cols` matrix. `out` receives `rows × out_cols` values.
///
/// # Safety
/// `data` is valid for `rows * cols` reads and `out` for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_robust_scaler_transform(
    h: *const TpmlRobustScaler,
    data: *const c_double,
    rows: usize,
    cols: usize,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        transform_into(m, data, rows, cols, out, out_len)
    })
}

/// Writes the number of values per output row into `out`.
///
/// # Safety
/// `h` is a live handle and `out` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn tpml_robust_scaler_out_cols(
    h: *const TpmlRobustScaler,
    out: *mut usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        output(out, 1, &[m.n_features().unwrap_or(0)])
    })
}

// ---------------------------------------------------------------- OneHotEncoder

/// Opaque one-hot encoder handle.
pub struct TpmlOneHotEncoder {
    fitted: Option<OneHotEncoder<f64>>,
}

/// Creates a one-hot encoder. Free it with `tpml_one_hot_encoder_free`.
#[no_mangle]
pub extern "C" fn tpml_one_hot_encoder_new() -> *mut TpmlOneHotEncoder {
    Box::into_raw(Box::new(TpmlOneHotEncoder { fitted: None }))
}

/// Frees a one-hot encoder handle. Null is ignored.
///
/// # Safety
/// `h` must be null or a live handle from `tpml_one_hot_encoder_new`.
#[no_mangle]
pub unsafe extern "C" fn tpml_one_hot_encoder_free(h: *mut TpmlOneHotEncoder) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Learns the categories of each column from a row-major `rows × cols` matrix.
///
/// # Safety
/// `data` is valid for `rows * cols` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_one_hot_encoder_fit(
    h: *mut TpmlOneHotEncoder,
    data: *const c_double,
    rows: usize,
    cols: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        let x = input(data, rows, cols)?;
        let mut m = OneHotEncoder::<f64>::new();
        UnsupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols)?)?;
        h.fitted = Some(m);
        Ok(())
    })
}

/// Encodes a row-major `rows × cols` matrix. `out` receives `rows × out_cols` indicators.
///
/// # Safety
/// `data` is valid for `rows * cols` reads and `out` for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_one_hot_encoder_transform(
    h: *const TpmlOneHotEncoder,
    data: *const c_double,
    rows: usize,
    cols: usize,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        transform_into(m, data, rows, cols, out, out_len)
    })
}

/// Writes the number of indicator columns into `out`: the total number of categories.
///
/// # Safety
/// `h` is a live handle and `out` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn tpml_one_hot_encoder_out_cols(
    h: *const TpmlOneHotEncoder,
    out: *mut usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        let width = m.categories().map_or(0, |c| c.iter().map(Vec::len).sum());
        output(out, 1, &[width])
    })
}

// ---------------------------------------------------------------- OrdinalEncoder

/// Opaque ordinal encoder handle.
pub struct TpmlOrdinalEncoder {
    fitted: Option<OrdinalEncoder<f64>>,
}

/// Creates an ordinal encoder. Free it with `tpml_ordinal_encoder_free`.
#[no_mangle]
pub extern "C" fn tpml_ordinal_encoder_new() -> *mut TpmlOrdinalEncoder {
    Box::into_raw(Box::new(TpmlOrdinalEncoder { fitted: None }))
}

/// Frees an ordinal encoder handle. Null is ignored.
///
/// # Safety
/// `h` must be null or a live handle from `tpml_ordinal_encoder_new`.
#[no_mangle]
pub unsafe extern "C" fn tpml_ordinal_encoder_free(h: *mut TpmlOrdinalEncoder) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Learns the categories of each column from a row-major `rows × cols` matrix.
///
/// # Safety
/// `data` is valid for `rows * cols` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_ordinal_encoder_fit(
    h: *mut TpmlOrdinalEncoder,
    data: *const c_double,
    rows: usize,
    cols: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        let x = input(data, rows, cols)?;
        let mut m = OrdinalEncoder::<f64>::new();
        UnsupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols)?)?;
        h.fitted = Some(m);
        Ok(())
    })
}

/// Encodes a row-major `rows × cols` matrix. `out` receives `rows × cols` indices.
///
/// # Safety
/// `data` is valid for `rows * cols` reads and `out` for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_ordinal_encoder_transform(
    h: *const TpmlOrdinalEncoder,
    data: *const c_double,
    rows: usize,
    cols: usize,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        transform_into(m, data, rows, cols, out, out_len)
    })
}

/// Writes the number of values per output row into `out`: one per input column.
///
/// # Safety
/// `h` is a live handle and `out` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn tpml_ordinal_encoder_out_cols(
    h: *const TpmlOrdinalEncoder,
    out: *mut usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        output(out, 1, &[m.categories().map_or(0, <[Vec<f64>]>::len)])
    })
}

// ---------------------------------------------------------------- SimpleImputer

/// Opaque imputer handle.
pub struct TpmlSimpleImputer {
    strategy: ImputeStrategy,
    fitted: Option<SimpleImputer<f64>>,
}

/// Creates an imputer. `fill_value` is used only by the `Constant` strategy. Free it with
/// `tpml_simple_imputer_free`.
#[no_mangle]
pub extern "C" fn tpml_simple_imputer_new(
    strategy: TpmlImputeStrategy,
    fill_value: c_double,
) -> *mut TpmlSimpleImputer {
    let strategy = match strategy {
        TpmlImputeStrategy::Mean => ImputeStrategy::Mean,
        TpmlImputeStrategy::Median => ImputeStrategy::Median,
        TpmlImputeStrategy::MostFrequent => ImputeStrategy::MostFrequent,
        TpmlImputeStrategy::Constant => {
            if !fill_value.is_finite() {
                return std::ptr::null_mut();
            }
            ImputeStrategy::Constant(fill_value)
        }
    };
    Box::into_raw(Box::new(TpmlSimpleImputer {
        strategy,
        fitted: None,
    }))
}

/// Frees an imputer handle. Null is ignored.
///
/// # Safety
/// `h` must be null or a live handle from `tpml_simple_imputer_new`.
#[no_mangle]
pub unsafe extern "C" fn tpml_simple_imputer_free(h: *mut TpmlSimpleImputer) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Learns the fill values from a row-major `rows × cols` matrix. NaN entries are ignored.
///
/// # Safety
/// `data` is valid for `rows * cols` reads.
#[no_mangle]
pub unsafe extern "C" fn tpml_simple_imputer_fit(
    h: *mut TpmlSimpleImputer,
    data: *const c_double,
    rows: usize,
    cols: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_mut().ok_or_else(null_handle)?;
        let x = input(data, rows, cols)?;
        let mut m = SimpleImputer::<f64>::new(h.strategy);
        UnsupervisedEstimator::fit(&mut m, DenseView::new(x, rows, cols)?)?;
        h.fitted = Some(m);
        Ok(())
    })
}

/// Replaces NaN entries of a row-major `rows × cols` matrix. `out` receives `rows × cols` values.
///
/// # Safety
/// `data` is valid for `rows * cols` reads and `out` for `out_len` writes.
#[no_mangle]
pub unsafe extern "C" fn tpml_simple_imputer_transform(
    h: *const TpmlSimpleImputer,
    data: *const c_double,
    rows: usize,
    cols: usize,
    out: *mut c_double,
    out_len: usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        transform_into(m, data, rows, cols, out, out_len)
    })
}

/// Writes the number of values per output row into `out`: the input width.
///
/// # Safety
/// `h` is a live handle and `out` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn tpml_simple_imputer_out_cols(
    h: *const TpmlSimpleImputer,
    out: *mut usize,
) -> TpmlStatus {
    run(|| {
        let h = h.as_ref().ok_or_else(null_handle)?;
        let m = h.fitted.as_ref().ok_or(Error::NotFitted)?;
        output(out, 1, &[m.statistics().map_or(0, <[f64]>::len)])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> Vec<f64> {
        vec![1.0, 10.0, 2.0, 30.0, 4.0, 10.0, 7.0, 50.0]
    }

    #[test]
    fn standard_scaler_round_trips_through_the_c_calls() {
        let x = data();
        let h = tpml_standard_scaler_new();
        assert!(!h.is_null());
        let mut width = 0usize;
        unsafe {
            assert_eq!(tpml_standard_scaler_fit(h, x.as_ptr(), 4, 2), TpmlStatus::Ok);
            assert_eq!(tpml_standard_scaler_out_cols(h, &mut width), TpmlStatus::Ok);
            assert_eq!(width, 2);
            let mut out = vec![0.0; 8];
            assert_eq!(
                tpml_standard_scaler_transform(h, x.as_ptr(), 4, 2, out.as_mut_ptr(), out.len()),
                TpmlStatus::Ok
            );
            let mean0: f64 = (0..4).map(|r| out[2 * r]).sum::<f64>() / 4.0;
            assert!(mean0.abs() < 1e-12);
            tpml_standard_scaler_free(h);
        }
    }

    #[test]
    fn transform_before_fit_is_not_fitted() {
        let x = data();
        let h = tpml_min_max_scaler_new();
        let mut out = vec![0.0; 8];
        let status = unsafe {
            tpml_min_max_scaler_transform(h, x.as_ptr(), 4, 2, out.as_mut_ptr(), out.len())
        };
        assert_eq!(status, TpmlStatus::NotFitted);
        unsafe { tpml_min_max_scaler_free(h) };
    }

    #[test]
    fn a_short_output_buffer_is_an_invalid_argument() {
        let x = data();
        let h = tpml_robust_scaler_new();
        unsafe {
            assert_eq!(tpml_robust_scaler_fit(h, x.as_ptr(), 4, 2), TpmlStatus::Ok);
            let mut out = vec![0.0; 3];
            assert_eq!(
                tpml_robust_scaler_transform(h, x.as_ptr(), 4, 2, out.as_mut_ptr(), out.len()),
                TpmlStatus::InvalidArgument
            );
            tpml_robust_scaler_free(h);
        }
    }

    #[test]
    fn the_one_hot_width_is_the_total_number_of_categories() {
        let cats = [0.0, 1.0, 1.0, 2.0, 0.0, 0.0];
        let h = tpml_one_hot_encoder_new();
        let mut width = 0usize;
        unsafe {
            assert_eq!(tpml_one_hot_encoder_fit(h, cats.as_ptr(), 3, 2), TpmlStatus::Ok);
            assert_eq!(tpml_one_hot_encoder_out_cols(h, &mut width), TpmlStatus::Ok);
            assert_eq!(width, 3 + 2);
            tpml_one_hot_encoder_free(h);
        }
    }

    #[test]
    fn an_imputer_with_a_constant_needs_a_finite_fill_value() {
        assert!(tpml_simple_imputer_new(TpmlImputeStrategy::Constant, f64::NAN).is_null());
        let h = tpml_simple_imputer_new(TpmlImputeStrategy::Constant, -9.0);
        assert!(!h.is_null());
        unsafe { tpml_simple_imputer_free(h) };
    }
}
