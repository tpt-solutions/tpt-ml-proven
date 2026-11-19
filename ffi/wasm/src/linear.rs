// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Linear models in WebAssembly: Ridge, Lasso, ElasticNet and logistic regression.

use wasm_bindgen::prelude::*;

use tpt_ml_proven_core::metrics::{accuracy, r2};
use tpt_ml_proven_core::ArrayView;
use tpt_ml_proven_core::{Predictor, SupervisedEstimator};
use tpt_ml_proven_linear::{ElasticNet, Lasso, LogisticRegression, MulticlassLogistic, Ridge};

use crate::options::Options;
use crate::{check_targets, js_err, not_fitted, view};

/// Ridge regression (closed form).
///
/// Options: `alpha` (1.0).
#[wasm_bindgen]
pub struct RidgeModel {
    alpha: f64,
    fitted: Option<Ridge<f64>>,
}

#[wasm_bindgen]
impl RidgeModel {
    #[wasm_bindgen(constructor)]
    pub fn new(options: Option<JsValue>) -> Result<RidgeModel, JsError> {
        let o = Options::new(options)?;
        Ok(RidgeModel {
            alpha: o.f64_or("alpha", 1.0)?,
            fitted: None,
        })
    }

    /// Fits to a flat row-major matrix and a target with one value per row.
    pub fn fit(&mut self, data: &[f64], cols: usize, y: &[f64]) -> Result<(), JsError> {
        let v = view(data, cols)?;
        check_targets(y, v.rows())?;
        let mut m = Ridge::<f64>::new(self.alpha);
        SupervisedEstimator::fit(&mut m, v, y).map_err(js_err)?;
        self.fitted = Some(m);
        Ok(())
    }

    /// Predicted target for each row.
    pub fn predict(&self, data: &[f64], cols: usize) -> Result<Vec<f64>, JsError> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Predictor::predict(m, view(data, cols)?).map_err(js_err)
    }

    /// R² of the predictions against `y`.
    pub fn score(&self, data: &[f64], cols: usize, y: &[f64]) -> Result<f64, JsError> {
        let pred = self.predict(data, cols)?;
        r2(y, &pred).map_err(js_err)
    }

    /// Fitted coefficients, one per feature.
    #[wasm_bindgen(getter)]
    pub fn coef(&self) -> Result<Vec<f64>, JsError> {
        Ok(self
            .fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .coef()
            .unwrap_or(&[])
            .to_vec())
    }

    /// Fitted intercept.
    #[wasm_bindgen(getter)]
    pub fn intercept(&self) -> Result<f64, JsError> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .intercept()
            .ok_or_else(not_fitted)
    }
}

/// Lasso regression (coordinate descent).
///
/// Options: `alpha` (1.0), `maxIter` (1000), `tol` (1e-4).
#[wasm_bindgen]
pub struct LassoModel {
    alpha: f64,
    max_iter: usize,
    tol: f64,
    fitted: Option<Lasso<f64>>,
}

#[wasm_bindgen]
impl LassoModel {
    #[wasm_bindgen(constructor)]
    pub fn new(options: Option<JsValue>) -> Result<LassoModel, JsError> {
        let o = Options::new(options)?;
        Ok(LassoModel {
            alpha: o.f64_or("alpha", 1.0)?,
            max_iter: o.usize_or("maxIter", 1000)?,
            tol: o.f64_or("tol", 1e-4)?,
            fitted: None,
        })
    }

    /// Fits to a flat row-major matrix and a target with one value per row.
    pub fn fit(&mut self, data: &[f64], cols: usize, y: &[f64]) -> Result<(), JsError> {
        let v = view(data, cols)?;
        check_targets(y, v.rows())?;
        let mut m = Lasso::<f64>::new(self.alpha)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol);
        SupervisedEstimator::fit(&mut m, v, y).map_err(js_err)?;
        self.fitted = Some(m);
        Ok(())
    }

    /// Predicted target for each row.
    pub fn predict(&self, data: &[f64], cols: usize) -> Result<Vec<f64>, JsError> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Predictor::predict(m, view(data, cols)?).map_err(js_err)
    }

    /// R² of the predictions against `y`.
    pub fn score(&self, data: &[f64], cols: usize, y: &[f64]) -> Result<f64, JsError> {
        let pred = self.predict(data, cols)?;
        r2(y, &pred).map_err(js_err)
    }

    /// Fitted coefficients, one per feature.
    #[wasm_bindgen(getter)]
    pub fn coef(&self) -> Result<Vec<f64>, JsError> {
        Ok(self
            .fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .coef()
            .unwrap_or(&[])
            .to_vec())
    }

    /// Fitted intercept.
    #[wasm_bindgen(getter)]
    pub fn intercept(&self) -> Result<f64, JsError> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .intercept()
            .ok_or_else(not_fitted)
    }
}

/// Elastic net regression (coordinate descent).
///
/// Options: `alpha` (1.0), `l1Ratio` (0.5), `maxIter` (1000), `tol` (1e-4).
#[wasm_bindgen]
pub struct ElasticNetModel {
    alpha: f64,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
    fitted: Option<ElasticNet<f64>>,
}

#[wasm_bindgen]
impl ElasticNetModel {
    #[wasm_bindgen(constructor)]
    pub fn new(options: Option<JsValue>) -> Result<ElasticNetModel, JsError> {
        let o = Options::new(options)?;
        Ok(ElasticNetModel {
            alpha: o.f64_or("alpha", 1.0)?,
            l1_ratio: o.f64_or("l1Ratio", 0.5)?,
            max_iter: o.usize_or("maxIter", 1000)?,
            tol: o.f64_or("tol", 1e-4)?,
            fitted: None,
        })
    }

    /// Fits to a flat row-major matrix and a target with one value per row.
    pub fn fit(&mut self, data: &[f64], cols: usize, y: &[f64]) -> Result<(), JsError> {
        let v = view(data, cols)?;
        check_targets(y, v.rows())?;
        let mut m = ElasticNet::<f64>::new(self.alpha, self.l1_ratio)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol);
        SupervisedEstimator::fit(&mut m, v, y).map_err(js_err)?;
        self.fitted = Some(m);
        Ok(())
    }

    /// Predicted target for each row.
    pub fn predict(&self, data: &[f64], cols: usize) -> Result<Vec<f64>, JsError> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Predictor::predict(m, view(data, cols)?).map_err(js_err)
    }

    /// R² of the predictions against `y`.
    pub fn score(&self, data: &[f64], cols: usize, y: &[f64]) -> Result<f64, JsError> {
        let pred = self.predict(data, cols)?;
        r2(y, &pred).map_err(js_err)
    }

    /// Fitted coefficients, one per feature.
    #[wasm_bindgen(getter)]
    pub fn coef(&self) -> Result<Vec<f64>, JsError> {
        Ok(self
            .fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .coef()
            .unwrap_or(&[])
            .to_vec())
    }

    /// Fitted intercept.
    #[wasm_bindgen(getter)]
    pub fn intercept(&self) -> Result<f64, JsError> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .intercept()
            .ok_or_else(not_fitted)
    }
}

enum LogitState {
    /// Two classes, mapped to 0 and 1; `classes` holds the original labels.
    Binary {
        model: LogisticRegression<f64>,
        classes: [f64; 2],
    },
    /// Three or more classes.
    Multi(MulticlassLogistic<f64>),
}

/// Logistic regression for two or more classes. Options: `C` (1.0, the inverse regularisation
/// strength), `maxIter` (100), `tol` (1e-4).
#[wasm_bindgen]
pub struct LogisticRegressionModel {
    c: f64,
    max_iter: usize,
    tol: f64,
    fitted: Option<LogitState>,
}

#[wasm_bindgen]
impl LogisticRegressionModel {
    #[wasm_bindgen(constructor)]
    pub fn new(options: Option<JsValue>) -> Result<LogisticRegressionModel, JsError> {
        let o = Options::new(options)?;
        Ok(LogisticRegressionModel {
            c: o.f64_or("C", 1.0)?,
            max_iter: o.usize_or("maxIter", 100)?,
            tol: o.f64_or("tol", 1e-4)?,
            fitted: None,
        })
    }

    /// Fits to a flat row-major matrix and class labels.
    pub fn fit(&mut self, data: &[f64], cols: usize, y: &[f64]) -> Result<(), JsError> {
        let v = view(data, cols)?;
        check_targets(y, v.rows())?;
        let mut classes = y.to_vec();
        classes.sort_by(f64::total_cmp);
        classes.dedup();
        let state = if classes.len() == 2 {
            let y01: Vec<f64> = y
                .iter()
                .map(|&t| if t == classes[1] { 1.0 } else { 0.0 })
                .collect();
            let mut m = LogisticRegression::<f64>::new(self.c)
                .with_max_iter(self.max_iter)
                .with_tol(self.tol);
            SupervisedEstimator::fit(&mut m, v, &y01[..]).map_err(js_err)?;
            LogitState::Binary {
                model: m,
                classes: [classes[0], classes[1]],
            }
        } else {
            let mut m = MulticlassLogistic::<f64>::new(self.c)
                .with_max_iter(self.max_iter)
                .with_tol(self.tol);
            SupervisedEstimator::fit(&mut m, v, y).map_err(js_err)?;
            LogitState::Multi(m)
        };
        self.fitted = Some(state);
        Ok(())
    }

    /// Predicted class label for each row.
    pub fn predict(&self, data: &[f64], cols: usize) -> Result<Vec<f64>, JsError> {
        let state = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let v = view(data, cols)?;
        match state {
            LogitState::Binary { model, classes } => {
                let zero_one = Predictor::predict(model, v).map_err(js_err)?;
                Ok(zero_one
                    .into_iter()
                    .map(|z| if z == 1.0 { classes[1] } else { classes[0] })
                    .collect())
            }
            LogitState::Multi(model) => Predictor::predict(model, v).map_err(js_err),
        }
    }

    /// Fraction of rows whose predicted label equals the label in `y`.
    pub fn score(&self, data: &[f64], cols: usize, y: &[f64]) -> Result<f64, JsError> {
        let pred = self.predict(data, cols)?;
        accuracy(y, &pred).map_err(js_err)
    }

    /// Class probabilities, flat with one row per sample and one column per class (in the
    /// order of `classes`).
    #[wasm_bindgen(js_name = predictProba)]
    pub fn predict_proba(&self, data: &[f64], cols: usize) -> Result<Vec<f64>, JsError> {
        let state = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let v = view(data, cols)?;
        match state {
            LogitState::Binary { model, .. } => {
                let p = model.predict_proba(v).map_err(js_err)?;
                Ok(p.iter().flat_map(|&p1| [1.0 - p1, p1]).collect())
            }
            LogitState::Multi(model) => model.predict_proba(v).map_err(js_err),
        }
    }

    /// The classes seen in training, sorted.
    #[wasm_bindgen(getter)]
    pub fn classes(&self) -> Result<Vec<f64>, JsError> {
        match self.fitted.as_ref().ok_or_else(not_fitted)? {
            LogitState::Binary { classes, .. } => Ok(classes.to_vec()),
            LogitState::Multi(model) => Ok(model.classes().unwrap_or(&[]).to_vec()),
        }
    }
}
