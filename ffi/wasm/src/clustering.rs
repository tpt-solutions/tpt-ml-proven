// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Clustering in WebAssembly: KMeans (in `f64` or `f32`), DBSCAN and HDBSCAN.

use wasm_bindgen::prelude::*;

use tpt_ml_proven_clustering::{Dbscan, Hdbscan, Init, KMeans};
use tpt_ml_proven_core::{DenseView, Float, Predictor, Transformer, UnsupervisedEstimator};

use crate::options::Options;
use crate::{flatten, js_err, not_fitted, peek_dtype, to_f32, view, view32};

enum KState {
    F32(KMeans<f32>),
    F64(KMeans<f64>),
}

/// KMeans clustering by Lloyd's algorithm.
///
/// Options: `nClusters` (required), `maxIter` (300), `tol` (1e-4), `init` (`"k-means++"` or
/// `"random"`), `seed` (0).
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
    #[wasm_bindgen(constructor)]
    pub fn new(options: Option<JsValue>) -> Result<KMeansModel, JsError> {
        let o = Options::new(options)?;
        let init = match o.str_or("init", "k-means++")?.as_str() {
            "k-means++" => Init::KMeansPlusPlus,
            "random" => Init::Random,
            other => {
                return Err(JsError::new(&format!(
                    "init must be 'k-means++' or 'random', not {other:?}"
                )))
            }
        };
        Ok(KMeansModel {
            n_clusters: o.required_usize("nClusters")?,
            max_iter: o.usize_or("maxIter", 300)?,
            tol: o.f64_or("tol", 1e-4)?,
            init,
            seed: o.seed_or("seed", 0)?,
            fitted: None,
        })
    }

    fn build<F: Float>(&self) -> KMeans<F> {
        KMeans::<F>::new(self.n_clusters)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_init(self.init)
            .with_seed(self.seed)
    }

    /// Fits to a flat row-major `Float64Array` with `cols` features per row.
    pub fn fit(&mut self, data: &[f64], cols: usize) -> Result<(), JsError> {
        let mut km = self.build::<f64>();
        UnsupervisedEstimator::fit(&mut km, view(data, cols)?).map_err(js_err)?;
        self.fitted = Some(KState::F64(km));
        Ok(())
    }

    /// As `fit`, for a `Float32Array`. The model is kept in single precision.
    #[wasm_bindgen(js_name = fitF32)]
    pub fn fit_f32(&mut self, data: &[f32], cols: usize) -> Result<(), JsError> {
        let mut km = self.build::<f32>();
        UnsupervisedEstimator::fit(&mut km, view32(data, cols)?).map_err(js_err)?;
        self.fitted = Some(KState::F32(km));
        Ok(())
    }

    /// Nearest cluster index for each row.
    pub fn predict(&self, data: &[f64], cols: usize) -> Result<Vec<u32>, JsError> {
        let labels: Vec<usize> = match self.fitted.as_ref().ok_or_else(not_fitted)? {
            KState::F64(km) => Predictor::predict(km, view(data, cols)?).map_err(js_err)?,
            KState::F32(km) => {
                let d = to_f32(data);
                Predictor::predict(km, view32(&d, cols)?).map_err(js_err)?
            }
        };
        Ok(labels.into_iter().map(|l| l as u32).collect())
    }

    /// Squared distance from each row to each centre, flat `rows × n_clusters`.
    pub fn transform(&self, data: &[f64], cols: usize) -> Result<Vec<f64>, JsError> {
        Ok(match self.fitted.as_ref().ok_or_else(not_fitted)? {
            KState::F64(km) => {
                flatten(&Transformer::transform(km, view(data, cols)?).map_err(js_err)?)
            }
            KState::F32(km) => {
                let d = to_f32(data);
                flatten(&Transformer::transform(km, view32(&d, cols)?).map_err(js_err)?)
            }
        })
    }

    /// Fitted centres, flat `n_clusters × n_features`.
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
        let labels: Vec<usize> = match self.fitted.as_ref().ok_or_else(not_fitted)? {
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
}

/// Density-based clustering. Options: `eps` and `minSamples` (both required).
#[wasm_bindgen]
pub struct DbscanModel {
    eps: f64,
    min_samples: usize,
    fitted: Option<Dbscan<f64>>,
}

#[wasm_bindgen]
impl DbscanModel {
    #[wasm_bindgen(constructor)]
    pub fn new(options: Option<JsValue>) -> Result<DbscanModel, JsError> {
        let o = Options::new(options)?;
        let eps = o
            .f64_opt("eps")?
            .ok_or_else(|| JsError::new("option eps is required"))?;
        Ok(DbscanModel {
            eps,
            min_samples: o.required_usize("minSamples")?,
            fitted: None,
        })
    }

    /// Fits to a flat row-major matrix.
    pub fn fit(&mut self, data: &[f64], cols: usize) -> Result<(), JsError> {
        let mut m = Dbscan::<f64>::new(self.eps, self.min_samples);
        UnsupervisedEstimator::fit(&mut m, view(data, cols)?).map_err(js_err)?;
        self.fitted = Some(m);
        Ok(())
    }

    /// Label of each training row, `-1` for noise.
    #[wasm_bindgen(getter)]
    pub fn labels(&self) -> Result<Vec<i32>, JsError> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(m.labels()
            .unwrap_or(&[])
            .iter()
            .map(|&l| l as i32)
            .collect())
    }

    /// Indices of the core samples.
    #[wasm_bindgen(getter, js_name = coreSampleIndices)]
    pub fn core_sample_indices(&self) -> Result<Vec<u32>, JsError> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(m.core_sample_indices()
            .unwrap_or(&[])
            .iter()
            .map(|&i| i as u32)
            .collect())
    }
}

/// Hierarchical density-based clustering. Options: `minClusterSize` (5), `minSamples`
/// (defaults to `minClusterSize`).
#[wasm_bindgen]
pub struct HdbscanModel {
    min_cluster_size: usize,
    min_samples: Option<usize>,
    fitted: Option<Hdbscan<f64>>,
}

#[wasm_bindgen]
impl HdbscanModel {
    #[wasm_bindgen(constructor)]
    pub fn new(options: Option<JsValue>) -> Result<HdbscanModel, JsError> {
        let o = Options::new(options)?;
        Ok(HdbscanModel {
            min_cluster_size: o.usize_or("minClusterSize", 5)?,
            min_samples: o.usize_opt("minSamples")?,
            fitted: None,
        })
    }

    /// Fits to a flat row-major matrix.
    pub fn fit(&mut self, data: &[f64], cols: usize) -> Result<(), JsError> {
        let mut m = Hdbscan::<f64>::new(self.min_cluster_size);
        if let Some(s) = self.min_samples {
            m = m.with_min_samples(s);
        }
        UnsupervisedEstimator::fit(&mut m, view(data, cols)?).map_err(js_err)?;
        self.fitted = Some(m);
        Ok(())
    }

    /// Label of each training row, `-1` for noise.
    #[wasm_bindgen(getter)]
    pub fn labels(&self) -> Result<Vec<i32>, JsError> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(m.labels()
            .unwrap_or(&[])
            .iter()
            .map(|&l| l as i32)
            .collect())
    }
}

#[allow(dead_code)]
fn _assert_dense_view_used(_: DenseView<'_, f64>) {}

#[wasm_bindgen]
impl KMeansModel {
    /// Serialises the fitted model as model-file bytes, for storage and later `fromBytes`.
    #[wasm_bindgen(js_name = toBytes)]
    pub fn to_bytes(&self) -> Result<Vec<u8>, JsError> {
        match self.fitted.as_ref().ok_or_else(not_fitted)? {
            KState::F64(km) => km.to_bytes().map_err(js_err),
            KState::F32(km) => km.to_bytes().map_err(js_err),
        }
    }

    /// Loads a model written by `toBytes`. The hyper-parameters are read from the file.
    #[wasm_bindgen(js_name = fromBytes)]
    pub fn from_bytes(bytes: &[u8]) -> Result<KMeansModel, JsError> {
        if peek_dtype(bytes)? == 1 {
            let km = KMeans::<f32>::from_bytes(bytes).map_err(js_err)?;
            Ok(KMeansModel {
                n_clusters: km.n_clusters(),
                max_iter: km.max_iter(),
                tol: km.tol(),
                init: km.init(),
                seed: km.seed(),
                fitted: Some(KState::F32(km)),
            })
        } else {
            let km = KMeans::<f64>::from_bytes(bytes).map_err(js_err)?;
            Ok(KMeansModel {
                n_clusters: km.n_clusters(),
                max_iter: km.max_iter(),
                tol: km.tol(),
                init: km.init(),
                seed: km.seed(),
                fitted: Some(KState::F64(km)),
            })
        }
    }
}

#[wasm_bindgen]
impl DbscanModel {
    /// Serialises the fitted model as model-file bytes.
    #[wasm_bindgen(js_name = toBytes)]
    pub fn to_bytes(&self) -> Result<Vec<u8>, JsError> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .to_bytes()
            .map_err(js_err)
    }

    /// Loads a model written by `toBytes`.
    #[wasm_bindgen(js_name = fromBytes)]
    pub fn from_bytes(bytes: &[u8]) -> Result<DbscanModel, JsError> {
        let m = Dbscan::<f64>::from_bytes(bytes).map_err(js_err)?;
        Ok(DbscanModel {
            eps: m.eps(),
            min_samples: m.min_samples(),
            fitted: Some(m),
        })
    }
}

#[wasm_bindgen]
impl HdbscanModel {
    /// Serialises the fitted model as model-file bytes.
    #[wasm_bindgen(js_name = toBytes)]
    pub fn to_bytes(&self) -> Result<Vec<u8>, JsError> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .to_bytes()
            .map_err(js_err)
    }

    /// Loads a model written by `toBytes`.
    #[wasm_bindgen(js_name = fromBytes)]
    pub fn from_bytes(bytes: &[u8]) -> Result<HdbscanModel, JsError> {
        let m = Hdbscan::<f64>::from_bytes(bytes).map_err(js_err)?;
        Ok(HdbscanModel {
            min_cluster_size: m.min_cluster_size(),
            min_samples: m.min_samples(),
            fitted: Some(m),
        })
    }
}
