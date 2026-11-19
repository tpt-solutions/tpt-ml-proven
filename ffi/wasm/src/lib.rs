// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! WebAssembly bindings for tpt-ml-proven.
//!
//! Every estimator takes one optional options object in its constructor, with camelCase keys
//! (see `options.rs`). Data is a flat, row-major typed array plus the number of columns:
//! `Float64Array` (or `Float32Array` for the `fitF32` methods), with `rows = length / cols`.
//! Matrices come back as flat arrays of `rows × cols` values. Labels are `Uint32Array` for
//! KMeans and `Int32Array` for the density clusterers (`-1` is noise), and class labels are
//! `Float64Array`s. Errors are thrown as JavaScript `Error` objects.
//!
//! Seeds are numbers, not BigInts, and must be integers no larger than `Number.MAX_SAFE_INTEGER`.

use wasm_bindgen::prelude::*;

use tpt_ml_proven_core::error::Error;
use tpt_ml_proven_core::{ArrayView, Dense, DenseView, Float};

mod clustering;
mod decomposition;
mod ensemble;
mod linear;
mod options;
mod preprocessing;
mod trees;

pub use clustering::{DbscanModel, HdbscanModel, KMeansModel};
pub use decomposition::{NmfModel, PcaModel, TruncatedSvdModel};
pub use ensemble::StumpForestModel;
pub use linear::{ElasticNetModel, LassoModel, LogisticRegressionModel, RidgeModel};
pub use preprocessing::{
    MinMaxScalerModel, OneHotEncoderModel, OrdinalEncoderModel, RobustScalerModel,
    SimpleImputerModel, StandardScalerModel,
};
pub use trees::{
    DecisionTreeClassifierModel, DecisionTreeRegressorModel, GradientBoostingRegressorModel,
    RandomForestClassifierModel, RandomForestRegressorModel,
};

/// Converts a library error into a JavaScript error.
pub(crate) fn js_err(e: Error) -> JsError {
    JsError::new(&e.to_string())
}

/// The error for a method called before `fit`.
pub(crate) fn not_fitted() -> JsError {
    JsError::new("this model is not fitted yet; call fit first")
}

/// Dtype of a model file: 1 for f32, 2 for f64. Version 1 files are f64.
pub(crate) fn peek_dtype(bytes: &[u8]) -> Result<u32, JsError> {
    if bytes.len() < 20 || bytes[..8] != *b"TPTMLMDL" {
        return Err(JsError::new("not a tpt-ml-proven model file"));
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes"));
    if version < 2 {
        return Ok(2);
    }
    Ok(u32::from_le_bytes(
        bytes[16..20].try_into().expect("4 bytes"),
    ))
}

/// Kind tag of a model file, which says which estimator wrote it.
pub(crate) fn peek_kind(bytes: &[u8]) -> Result<u32, JsError> {
    bytes
        .get(12..16)
        .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
        .ok_or_else(|| JsError::new("not a tpt-ml-proven model file"))
}

/// Number of rows in a flat row-major array of `cols` features per row.
pub(crate) fn check_shape(len: usize, cols: usize) -> Result<usize, JsError> {
    if cols == 0 {
        return Err(JsError::new("cols must be at least 1"));
    }
    if len % cols != 0 {
        return Err(JsError::new("data length is not a multiple of cols"));
    }
    Ok(len / cols)
}

/// A borrowed view of a flat `f64` matrix.
pub(crate) fn view(data: &[f64], cols: usize) -> Result<DenseView<'_, f64>, JsError> {
    let rows = check_shape(data.len(), cols)?;
    DenseView::new(data, rows, cols).map_err(|e| JsError::new(&e.to_string()))
}

/// A borrowed view of a flat `f32` matrix.
pub(crate) fn view32(data: &[f32], cols: usize) -> Result<DenseView<'_, f32>, JsError> {
    let rows = check_shape(data.len(), cols)?;
    DenseView::new(data, rows, cols).map_err(|e| JsError::new(&e.to_string()))
}

/// The number of targets must equal the number of rows.
pub(crate) fn check_targets(y: &[f64], rows: usize) -> Result<(), JsError> {
    if y.len() != rows {
        return Err(JsError::new(
            "the target has a different length from the number of rows of X",
        ));
    }
    Ok(())
}

/// A matrix as a flat row-major `Vec<f64>`.
pub(crate) fn flatten<F: Float>(d: &Dense<F>) -> Vec<f64> {
    let mut out = Vec::with_capacity(d.rows() * d.cols());
    for r in 0..d.rows() {
        for c in 0..d.cols() {
            out.push(d.get(r, c).to_f64());
        }
    }
    out
}

/// Narrows a flat `f64` matrix to `f32`, for the `f32` models.
pub(crate) fn to_f32(data: &[f64]) -> Vec<f32> {
    data.iter().map(|&v| v as f32).collect()
}

/// The version of the library, as a string.
#[wasm_bindgen(js_name = version)]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
