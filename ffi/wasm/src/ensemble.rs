// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Ensembles in WebAssembly: the stump forest.

use wasm_bindgen::prelude::*;

use tpt_ml_proven_core::metrics::accuracy;
use tpt_ml_proven_core::ArrayView;
use tpt_ml_proven_core::{Predictor, SupervisedEstimator};
use tpt_ml_proven_ensemble::StumpForest;

use crate::options::Options;
use crate::{check_targets, js_err, not_fitted, view};

/// Bagged depth-one classification trees with a majority vote.
///
/// Options: `nEstimators` (100), `seed` (0).
#[wasm_bindgen]
pub struct StumpForestModel {
    n_estimators: usize,
    seed: u64,
    fitted: Option<StumpForest<f64>>,
}

#[wasm_bindgen]
impl StumpForestModel {
    #[wasm_bindgen(constructor)]
    pub fn new(options: Option<JsValue>) -> Result<StumpForestModel, JsError> {
        let o = Options::new(options)?;
        Ok(StumpForestModel {
            n_estimators: o.usize_or("nEstimators", 100)?,
            seed: o.seed_or("seed", 0)?,
            fitted: None,
        })
    }

    /// Fits to a flat row-major matrix and class labels.
    pub fn fit(&mut self, data: &[f64], cols: usize, y: &[f64]) -> Result<(), JsError> {
        let v = view(data, cols)?;
        check_targets(y, v.rows())?;
        let mut m = StumpForest::<f64>::new(self.n_estimators).with_seed(self.seed);
        SupervisedEstimator::fit(&mut m, v, y).map_err(js_err)?;
        self.fitted = Some(m);
        Ok(())
    }

    /// Predicted class label for each row.
    pub fn predict(&self, data: &[f64], cols: usize) -> Result<Vec<f64>, JsError> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Predictor::predict(m, view(data, cols)?).map_err(js_err)
    }

    /// Fraction of rows whose predicted label equals the label in `y`.
    pub fn score(&self, data: &[f64], cols: usize, y: &[f64]) -> Result<f64, JsError> {
        let pred = self.predict(data, cols)?;
        accuracy(y, &pred).map_err(js_err)
    }
}

#[wasm_bindgen]
impl StumpForestModel {
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
    pub fn from_bytes(bytes: &[u8]) -> Result<StumpForestModel, JsError> {
        let m = StumpForest::<f64>::from_bytes(bytes).map_err(js_err)?;
        Ok(StumpForestModel {
            n_estimators: m.n_estimators_setting(),
            seed: m.seed(),
            fitted: Some(m),
        })
    }
}
