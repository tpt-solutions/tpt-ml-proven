// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Matrix decompositions in WebAssembly: PCA (in `f64` or `f32`), truncated SVD and NMF.

use wasm_bindgen::prelude::*;

use tpt_ml_proven_core::{DenseView, Transformer, UnsupervisedEstimator};
use tpt_ml_proven_decomposition::{Nmf, Pca, TruncatedSvd};

use crate::options::Options;
use crate::{check_shape, flatten, js_err, not_fitted, to_f32, view, view32};

enum PState {
    F32(Pca<f32>),
    F64(Pca<f64>),
}

/// Principal component analysis, optionally in `f32`.
///
/// Options: `nComponents` (required).
#[wasm_bindgen]
pub struct PcaModel {
    n_components: usize,
    fitted: Option<PState>,
}

#[wasm_bindgen]
impl PcaModel {
    #[wasm_bindgen(constructor)]
    pub fn new(options: Option<JsValue>) -> Result<PcaModel, JsError> {
        let o = Options::new(options)?;
        Ok(PcaModel {
            n_components: o.required_usize("nComponents")?,
            fitted: None,
        })
    }

    /// Fits to a flat row-major `Float64Array` with `cols` features per row.
    pub fn fit(&mut self, data: &[f64], cols: usize) -> Result<(), JsError> {
        let mut p = Pca::<f64>::new(self.n_components);
        UnsupervisedEstimator::fit(&mut p, view(data, cols)?).map_err(js_err)?;
        self.fitted = Some(PState::F64(p));
        Ok(())
    }

    /// As `fit`, for a `Float32Array`. The model is kept in single precision.
    #[wasm_bindgen(js_name = fitF32)]
    pub fn fit_f32(&mut self, data: &[f32], cols: usize) -> Result<(), JsError> {
        let mut p = Pca::<f32>::new(self.n_components);
        UnsupervisedEstimator::fit(&mut p, view32(data, cols)?).map_err(js_err)?;
        self.fitted = Some(PState::F32(p));
        Ok(())
    }

    /// Projection onto the kept components, flat `rows × n_components`.
    pub fn transform(&self, data: &[f64], cols: usize) -> Result<Vec<f64>, JsError> {
        Ok(match self.fitted.as_ref().ok_or_else(not_fitted)? {
            PState::F64(p) => {
                flatten(&Transformer::transform(p, view(data, cols)?).map_err(js_err)?)
            }
            PState::F32(p) => {
                let d = to_f32(data);
                flatten(&Transformer::transform(p, view32(&d, cols)?).map_err(js_err)?)
            }
        })
    }

    /// Maps component scores back to feature space, flat `rows × n_features`.
    #[wasm_bindgen(js_name = inverseTransform)]
    pub fn inverse_transform(
        &self,
        scores: &[f64],
        n_components: usize,
    ) -> Result<Vec<f64>, JsError> {
        let rows = check_shape(scores.len(), n_components)?;
        let z =
            DenseView::new(scores, rows, n_components).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(match self.fitted.as_ref().ok_or_else(not_fitted)? {
            PState::F64(p) => flatten(&p.inverse_transform(&z).map_err(js_err)?),
            PState::F32(p) => {
                let d = to_f32(scores);
                let z32 = DenseView::new(&d, rows, n_components)
                    .map_err(|e| JsError::new(&e.to_string()))?;
                flatten(&p.inverse_transform(&z32).map_err(js_err)?)
            }
        })
    }

    /// Principal axes, flat `n_components × n_features`.
    #[wasm_bindgen(getter)]
    pub fn components(&self) -> Result<Vec<f64>, JsError> {
        match self.fitted.as_ref().ok_or_else(not_fitted)? {
            PState::F64(p) => Ok(flatten(p.components().ok_or_else(not_fitted)?)),
            PState::F32(p) => Ok(flatten(p.components().ok_or_else(not_fitted)?)),
        }
    }

    /// Per-feature mean removed before projection.
    #[wasm_bindgen(getter)]
    pub fn mean(&self) -> Result<Vec<f64>, JsError> {
        match self.fitted.as_ref().ok_or_else(not_fitted)? {
            PState::F64(p) => Ok(p.mean().ok_or_else(not_fitted)?.to_vec()),
            PState::F32(p) => Ok(p
                .mean()
                .ok_or_else(not_fitted)?
                .iter()
                .map(|v| *v as f64)
                .collect()),
        }
    }

    /// Variance along each kept component.
    #[wasm_bindgen(getter, js_name = explainedVariance)]
    pub fn explained_variance(&self) -> Result<Vec<f64>, JsError> {
        match self.fitted.as_ref().ok_or_else(not_fitted)? {
            PState::F64(p) => Ok(p.explained_variance().ok_or_else(not_fitted)?.to_vec()),
            PState::F32(p) => Ok(p
                .explained_variance()
                .ok_or_else(not_fitted)?
                .iter()
                .map(|v| *v as f64)
                .collect()),
        }
    }

    /// Fraction of total variance explained by each kept component.
    #[wasm_bindgen(getter, js_name = explainedVarianceRatio)]
    pub fn explained_variance_ratio(&self) -> Result<Vec<f64>, JsError> {
        match self.fitted.as_ref().ok_or_else(not_fitted)? {
            PState::F64(p) => Ok(p
                .explained_variance_ratio()
                .ok_or_else(not_fitted)?
                .to_vec()),
            PState::F32(p) => Ok(p
                .explained_variance_ratio()
                .ok_or_else(not_fitted)?
                .iter()
                .map(|v| *v as f64)
                .collect()),
        }
    }
}

/// Truncated singular value decomposition on uncentred data.
///
/// Options: `nComponents` (required).
#[wasm_bindgen]
pub struct TruncatedSvdModel {
    n_components: usize,
    fitted: Option<TruncatedSvd<f64>>,
}

#[wasm_bindgen]
impl TruncatedSvdModel {
    #[wasm_bindgen(constructor)]
    pub fn new(options: Option<JsValue>) -> Result<TruncatedSvdModel, JsError> {
        let o = Options::new(options)?;
        Ok(TruncatedSvdModel {
            n_components: o.required_usize("nComponents")?,
            fitted: None,
        })
    }

    /// Fits to a flat row-major matrix.
    pub fn fit(&mut self, data: &[f64], cols: usize) -> Result<(), JsError> {
        let mut m = TruncatedSvd::<f64>::new(self.n_components);
        UnsupervisedEstimator::fit(&mut m, view(data, cols)?).map_err(js_err)?;
        self.fitted = Some(m);
        Ok(())
    }

    /// Projection onto the right singular vectors, flat `rows × n_components`.
    pub fn transform(&self, data: &[f64], cols: usize) -> Result<Vec<f64>, JsError> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(flatten(
            &Transformer::transform(m, view(data, cols)?).map_err(js_err)?,
        ))
    }

    /// Singular values, largest first.
    #[wasm_bindgen(getter, js_name = singularValues)]
    pub fn singular_values(&self) -> Result<Vec<f64>, JsError> {
        Ok(self
            .fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .singular_values()
            .unwrap_or(&[])
            .to_vec())
    }

    /// Right singular vectors, flat `n_components × n_features`.
    #[wasm_bindgen(getter)]
    pub fn components(&self) -> Result<Vec<f64>, JsError> {
        Ok(flatten(
            self.fitted
                .as_ref()
                .ok_or_else(not_fitted)?
                .components()
                .ok_or_else(not_fitted)?,
        ))
    }
}

/// Non-negative matrix factorisation with multiplicative updates.
///
/// Options: `nComponents` (required), `maxIter` (200), `tol` (1e-4), `seed` (0).
#[wasm_bindgen]
pub struct NmfModel {
    n_components: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    fitted: Option<Nmf<f64>>,
}

#[wasm_bindgen]
impl NmfModel {
    #[wasm_bindgen(constructor)]
    pub fn new(options: Option<JsValue>) -> Result<NmfModel, JsError> {
        let o = Options::new(options)?;
        Ok(NmfModel {
            n_components: o.required_usize("nComponents")?,
            max_iter: o.usize_or("maxIter", 200)?,
            tol: o.f64_or("tol", 1e-4)?,
            seed: o.seed_or("seed", 0)?,
            fitted: None,
        })
    }

    /// Fits to a flat row-major non-negative matrix.
    pub fn fit(&mut self, data: &[f64], cols: usize) -> Result<(), JsError> {
        let mut m = Nmf::<f64>::new(self.n_components)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_seed(self.seed);
        UnsupervisedEstimator::fit(&mut m, view(data, cols)?).map_err(js_err)?;
        self.fitted = Some(m);
        Ok(())
    }

    /// The `W` factors for `data` with the learned `H` held fixed, flat `rows × n_components`.
    pub fn transform(&self, data: &[f64], cols: usize) -> Result<Vec<f64>, JsError> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(flatten(
            &Transformer::transform(m, view(data, cols)?).map_err(js_err)?,
        ))
    }

    /// Learned factors `H`, flat `n_components × n_features`.
    #[wasm_bindgen(getter)]
    pub fn components(&self) -> Result<Vec<f64>, JsError> {
        Ok(flatten(
            self.fitted
                .as_ref()
                .ok_or_else(not_fitted)?
                .components()
                .ok_or_else(not_fitted)?,
        ))
    }

    /// Multiplicative-update iterations used.
    #[wasm_bindgen(getter, js_name = nIter)]
    pub fn n_iter(&self) -> Result<usize, JsError> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .n_iter()
            .ok_or_else(not_fitted)
    }
}
