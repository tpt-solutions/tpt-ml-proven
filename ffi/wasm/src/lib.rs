// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! WebAssembly bindings for tpt-ml-proven: `KMeans` and `PCA`.
//!
//! Data is passed as a flat, row-major typed array plus the number of columns:
//! `Float64Array` or `Float32Array`, with `rows = length / cols`. Results come back as
//! flat typed arrays, and `rows`/`cols` are given by the methods' documented shapes.
//! Errors are thrown as JavaScript `Error` objects.

use wasm_bindgen::prelude::*;

use tpt_ml_proven_clustering::{Init, KMeans};
use tpt_ml_proven_core::error::Error;
use tpt_ml_proven_core::{
    ArrayView, Dense, DenseView, Float, Predictor, Transformer, UnsupervisedEstimator,
};
use tpt_ml_proven_decomposition::Pca;

mod models;

pub use models::{
    DbscanModel, ElasticNetModel, HdbscanModel, LassoModel, LogisticRegressionModel, NmfModel,
    RidgeModel, StumpForestModel, TruncatedSvdModel,
};

fn js_err(e: Error) -> JsError {
    JsError::new(&e.to_string())
}

/// Number of rows in a flat row-major array of `cols` features per row.
fn check_shape(len: usize, cols: usize) -> Result<usize, JsError> {
    if cols == 0 {
        return Err(JsError::new("cols must be at least 1"));
    }
    if len % cols != 0 {
        return Err(JsError::new("data length is not a multiple of cols"));
    }
    Ok(len / cols)
}

fn flatten<F: Float>(d: &Dense<F>) -> Vec<f64> {
    let mut out = Vec::with_capacity(d.rows() * d.cols());
    for r in 0..d.rows() {
        for c in 0..d.cols() {
            out.push(d.get(r, c).to_f64());
        }
    }
    out
}

fn to_f32(data: &[f64]) -> Vec<f32> {
    data.iter().map(|&v| v as f32).collect()
}

// ---------------------------------------------------------------- KMeans

enum KState {
    F32(KMeans<f32>),
    F64(KMeans<f64>),
}

/// KMeans clustering by Lloyd's algorithm.
#[wasm_bindgen]
pub struct KMeansModel {
    n_clusters: usize,
    max_iter: usize,
    tol: f64,
    init: Init,
    seed: u64,
    fitted: Option<KState>,
}

#[wasm_bindgen]
impl KMeansModel {
    /// `init` is `"k-means++"` (default) or `"random"`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        n_clusters: usize,
        max_iter: Option<usize>,
        tol: Option<f64>,
        init: Option<String>,
        seed: Option<u64>,
    ) -> Result<KMeansModel, JsError> {
        let init = match init.as_deref().unwrap_or("k-means++") {
            "k-means++" => Init::KMeansPlusPlus,
            "random" => Init::Random,
            other => {
                return Err(JsError::new(&format!(
                    "init must be 'k-means++' or 'random', not {other:?}"
                )))
            }
        };
        Ok(KMeansModel {
            n_clusters,
            max_iter: max_iter.unwrap_or(300),
            tol: tol.unwrap_or(1e-4),
            init,
            seed: seed.unwrap_or(0),
            fitted: None,
        })
    }

    /// Fits to a flat row-major `Float64Array` with `cols` features per row.
    pub fn fit(&mut self, data: &[f64], cols: usize) -> Result<(), JsError> {
        let rows = check_shape(data.len(), cols)?;
        let mut km = self.build::<f64>();
        UnsupervisedEstimator::fit(
            &mut km,
            DenseView::new(data, rows, cols).map_err(|e| js_err(e.into()))?,
        )
        .map_err(js_err)?;
        self.fitted = Some(KState::F64(km));
        Ok(())
    }

    /// As `fit`, for a `Float32Array`. The model is kept in single precision.
    #[wasm_bindgen(js_name = fitF32)]
    pub fn fit_f32(&mut self, data: &[f32], cols: usize) -> Result<(), JsError> {
        let rows = check_shape(data.len(), cols)?;
        let mut km = self.build::<f32>();
        UnsupervisedEstimator::fit(
            &mut km,
            DenseView::new(data, rows, cols).map_err(|e| js_err(e.into()))?,
        )
        .map_err(js_err)?;
        self.fitted = Some(KState::F32(km));
        Ok(())
    }

    /// Nearest cluster index for each row.
    pub fn predict(&self, data: &[f64], cols: usize) -> Result<Vec<u32>, JsError> {
        let rows = check_shape(data.len(), cols)?;
        let labels = match self.fitted.as_ref().ok_or_else(not_fitted)? {
            KState::F64(km) => km
                .predict(DenseView::new(data, rows, cols).map_err(|e| js_err(e.into()))?)
                .map_err(js_err)?,
            KState::F32(km) => {
                let d = to_f32(data);
                km.predict(DenseView::new(&d, rows, cols).map_err(|e| js_err(e.into()))?)
                    .map_err(js_err)?
            }
        };
        Ok(labels.into_iter().map(|l| l as u32).collect())
    }

    /// Squared distance from each row to each centre, flat `rows x n_clusters`.
    pub fn transform(&self, data: &[f64], cols: usize) -> Result<Vec<f64>, JsError> {
        let rows = check_shape(data.len(), cols)?;
        Ok(match self.fitted.as_ref().ok_or_else(not_fitted)? {
            KState::F64(km) => flatten(
                &km.transform(DenseView::new(data, rows, cols).map_err(|e| js_err(e.into()))?)
                    .map_err(js_err)?,
            ),
            KState::F32(km) => {
                let d = to_f32(data);
                flatten(
                    &km.transform(DenseView::new(&d, rows, cols).map_err(|e| js_err(e.into()))?)
                        .map_err(js_err)?,
                )
            }
        })
    }

    /// Fitted centres, flat `n_clusters x n_features`.
    #[wasm_bindgen(getter)]
    pub fn centers(&self) -> Result<Vec<f64>, JsError> {
        match self.fitted.as_ref().ok_or_else(not_fitted)? {
            KState::F64(km) => Ok(flatten(km.cluster_centers().ok_or_else(not_fitted)?)),
            KState::F32(km) => Ok(flatten(km.cluster_centers().ok_or_else(not_fitted)?)),
        }
    }

    /// Cluster label of each training row.
    #[wasm_bindgen(getter)]
    pub fn labels(&self) -> Result<Vec<u32>, JsError> {
        let labels = match self.fitted.as_ref().ok_or_else(not_fitted)? {
            KState::F64(km) => km.labels().unwrap_or(&[]).to_vec(),
            KState::F32(km) => km.labels().unwrap_or(&[]).to_vec(),
        };
        Ok(labels.into_iter().map(|l| l as u32).collect())
    }

    /// Sum of squared distances from each training row to its centre.
    #[wasm_bindgen(getter)]
    pub fn inertia(&self) -> Result<f64, JsError> {
        match self.fitted.as_ref().ok_or_else(not_fitted)? {
            KState::F64(km) => km.inertia().ok_or_else(not_fitted),
            KState::F32(km) => Ok(km.inertia().ok_or_else(not_fitted)?.to_f64()),
        }
    }

    /// Lloyd iterations performed.
    #[wasm_bindgen(getter, js_name = nIter)]
    pub fn n_iter(&self) -> Result<usize, JsError> {
        match self.fitted.as_ref().ok_or_else(not_fitted)? {
            KState::F64(km) => km.n_iter().ok_or_else(not_fitted),
            KState::F32(km) => km.n_iter().ok_or_else(not_fitted),
        }
    }

    fn build<F: Float>(&self) -> KMeans<F> {
        KMeans::<F>::new(self.n_clusters)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_init(self.init)
            .with_seed(self.seed)
    }
}

fn not_fitted() -> JsError {
    JsError::new("this model is not fitted yet; call fit first")
}

// ---------------------------------------------------------------- PCA

enum PState {
    F32(Pca<f32>),
    F64(Pca<f64>),
}

/// Principal component analysis.
#[wasm_bindgen]
pub struct PcaModel {
    n_components: usize,
    fitted: Option<PState>,
}

#[wasm_bindgen]
impl PcaModel {
    #[wasm_bindgen(constructor)]
    pub fn new(n_components: usize) -> PcaModel {
        PcaModel {
            n_components,
            fitted: None,
        }
    }

    /// Fits to a flat row-major `Float64Array` with `cols` features per row.
    pub fn fit(&mut self, data: &[f64], cols: usize) -> Result<(), JsError> {
        let rows = check_shape(data.len(), cols)?;
        let mut p = Pca::<f64>::new(self.n_components);
        UnsupervisedEstimator::fit(
            &mut p,
            DenseView::new(data, rows, cols).map_err(|e| js_err(e.into()))?,
        )
        .map_err(js_err)?;
        self.fitted = Some(PState::F64(p));
        Ok(())
    }

    /// As `fit`, for a `Float32Array`.
    #[wasm_bindgen(js_name = fitF32)]
    pub fn fit_f32(&mut self, data: &[f32], cols: usize) -> Result<(), JsError> {
        let rows = check_shape(data.len(), cols)?;
        let mut p = Pca::<f32>::new(self.n_components);
        UnsupervisedEstimator::fit(
            &mut p,
            DenseView::new(data, rows, cols).map_err(|e| js_err(e.into()))?,
        )
        .map_err(js_err)?;
        self.fitted = Some(PState::F32(p));
        Ok(())
    }

    /// Projection onto the kept components, flat `rows x n_components`.
    pub fn transform(&self, data: &[f64], cols: usize) -> Result<Vec<f64>, JsError> {
        let rows = check_shape(data.len(), cols)?;
        Ok(match self.fitted.as_ref().ok_or_else(not_fitted)? {
            PState::F64(p) => flatten(
                &p.transform(DenseView::new(data, rows, cols).map_err(|e| js_err(e.into()))?)
                    .map_err(js_err)?,
            ),
            PState::F32(p) => {
                let d = to_f32(data);
                flatten(
                    &p.transform(DenseView::new(&d, rows, cols).map_err(|e| js_err(e.into()))?)
                        .map_err(js_err)?,
                )
            }
        })
    }

    /// Maps component scores back to feature space, flat `rows x n_features`.
    #[wasm_bindgen(js_name = inverseTransform)]
    pub fn inverse_transform(
        &self,
        scores: &[f64],
        n_components: usize,
    ) -> Result<Vec<f64>, JsError> {
        let rows = check_shape(scores.len(), n_components)?;
        Ok(match self.fitted.as_ref().ok_or_else(not_fitted)? {
            PState::F64(p) => flatten(
                &p.inverse_transform(
                    &DenseView::new(scores, rows, n_components).map_err(|e| js_err(e.into()))?,
                )
                .map_err(js_err)?,
            ),
            PState::F32(p) => {
                let d = to_f32(scores);
                flatten(
                    &p.inverse_transform(
                        &DenseView::new(&d, rows, n_components).map_err(|e| js_err(e.into()))?,
                    )
                    .map_err(js_err)?,
                )
            }
        })
    }

    /// Principal axes, flat `n_components x n_features`.
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
                .map(|v| v.to_f64())
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
                .map(|v| v.to_f64())
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
                .map(|v| v.to_f64())
                .collect()),
        }
    }
}
