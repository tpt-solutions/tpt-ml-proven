// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Trees and forests in WebAssembly: single trees, gradient boosting and random forests, for
//! regression and classification.

use wasm_bindgen::prelude::*;

use tpt_ml_proven_core::metrics::{accuracy, r2};
use tpt_ml_proven_core::ArrayView;
use tpt_ml_proven_core::{Predictor, SupervisedEstimator};
use tpt_ml_proven_trees::{
    Criterion, DecisionTreeClassifier, DecisionTreeRegressor, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
};

use crate::options::Options;
use crate::{check_targets, js_err, not_fitted, view};

fn parse_criterion(text: &str) -> Result<Criterion, JsError> {
    match text {
        "gini" => Ok(Criterion::Gini),
        "entropy" => Ok(Criterion::Entropy),
        other => Err(JsError::new(&format!(
            "criterion must be 'gini' or 'entropy', not {other:?}"
        ))),
    }
}

/// A single CART regression tree.
///
/// Options: `maxDepth` (3), `minSamplesLeaf` (1).
#[wasm_bindgen]
pub struct DecisionTreeRegressorModel {
    max_depth: usize,
    min_samples_leaf: usize,
    fitted: Option<DecisionTreeRegressor>,
}

#[wasm_bindgen]
impl DecisionTreeRegressorModel {
    #[wasm_bindgen(constructor)]
    pub fn new(options: Option<JsValue>) -> Result<DecisionTreeRegressorModel, JsError> {
        let o = Options::new(options)?;
        Ok(DecisionTreeRegressorModel {
            max_depth: o.usize_or("maxDepth", 3)?,
            min_samples_leaf: o.usize_or("minSamplesLeaf", 1)?,
            fitted: None,
        })
    }

    /// Fits to a flat row-major matrix and a target with one value per row.
    pub fn fit(&mut self, data: &[f64], cols: usize, y: &[f64]) -> Result<(), JsError> {
        let v = view(data, cols)?;
        check_targets(y, v.rows())?;
        let mut m =
            DecisionTreeRegressor::new(self.max_depth).with_min_samples_leaf(self.min_samples_leaf);
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
}

/// A single CART classification tree.
///
/// Options: `maxDepth` (3), `minSamplesLeaf` (1), `criterion` (`"gini"` or `"entropy"`).
#[wasm_bindgen]
pub struct DecisionTreeClassifierModel {
    max_depth: usize,
    min_samples_leaf: usize,
    criterion: Criterion,
    fitted: Option<DecisionTreeClassifier>,
}

#[wasm_bindgen]
impl DecisionTreeClassifierModel {
    #[wasm_bindgen(constructor)]
    pub fn new(options: Option<JsValue>) -> Result<DecisionTreeClassifierModel, JsError> {
        let o = Options::new(options)?;
        Ok(DecisionTreeClassifierModel {
            max_depth: o.usize_or("maxDepth", 3)?,
            min_samples_leaf: o.usize_or("minSamplesLeaf", 1)?,
            criterion: parse_criterion(&o.str_or("criterion", "gini")?)?,
            fitted: None,
        })
    }

    /// Fits to a flat row-major matrix and class labels.
    pub fn fit(&mut self, data: &[f64], cols: usize, y: &[f64]) -> Result<(), JsError> {
        let v = view(data, cols)?;
        check_targets(y, v.rows())?;
        let mut m = DecisionTreeClassifier::new(self.max_depth)
            .with_min_samples_leaf(self.min_samples_leaf)
            .with_criterion(self.criterion);
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

    /// The classes seen in training, sorted.
    #[wasm_bindgen(getter)]
    pub fn classes(&self) -> Result<Vec<f64>, JsError> {
        Ok(self
            .fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .classes()
            .unwrap_or(&[])
            .to_vec())
    }
}

/// Gradient boosting for regression, with squared-error loss.
///
/// Options: `nEstimators` (100), `learningRate` (0.1), `maxDepth` (3), `minSamplesLeaf` (1),
/// `subsample` (1.0: every row each round), `maxFeatures` (all features), `seed` (0).
#[wasm_bindgen]
pub struct GradientBoostingRegressorModel {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_leaf: usize,
    subsample: f64,
    max_features: Option<usize>,
    seed: u64,
    fitted: Option<GradientBoostingRegressor>,
}

#[wasm_bindgen]
impl GradientBoostingRegressorModel {
    #[wasm_bindgen(constructor)]
    pub fn new(options: Option<JsValue>) -> Result<GradientBoostingRegressorModel, JsError> {
        let o = Options::new(options)?;
        Ok(GradientBoostingRegressorModel {
            n_estimators: o.usize_or("nEstimators", 100)?,
            learning_rate: o.f64_or("learningRate", 0.1)?,
            max_depth: o.usize_or("maxDepth", 3)?,
            min_samples_leaf: o.usize_or("minSamplesLeaf", 1)?,
            subsample: o.f64_or("subsample", 1.0)?,
            max_features: o.usize_opt("maxFeatures")?,
            seed: o.seed_or("seed", 0)?,
            fitted: None,
        })
    }

    /// Fits to a flat row-major matrix and a target with one value per row.
    pub fn fit(&mut self, data: &[f64], cols: usize, y: &[f64]) -> Result<(), JsError> {
        let v = view(data, cols)?;
        check_targets(y, v.rows())?;
        let mut m = GradientBoostingRegressor::new(self.n_estimators, self.max_depth)
            .with_learning_rate(self.learning_rate)
            .with_min_samples_leaf(self.min_samples_leaf)
            .with_subsample(self.subsample)
            .with_seed(self.seed);
        if let Some(f) = self.max_features {
            m = m.with_max_features(f);
        }
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
}

/// A random forest of regression trees: bootstrap samples, a random feature subset per split,
/// and the mean of the trees' predictions.
///
/// Options: `nEstimators` (100), `maxDepth` (10), `minSamplesLeaf` (1), `maxFeatures` (all
/// features), `seed` (0).
#[wasm_bindgen]
pub struct RandomForestRegressorModel {
    n_estimators: usize,
    max_depth: usize,
    min_samples_leaf: usize,
    max_features: Option<usize>,
    seed: u64,
    fitted: Option<RandomForestRegressor>,
}

#[wasm_bindgen]
impl RandomForestRegressorModel {
    #[wasm_bindgen(constructor)]
    pub fn new(options: Option<JsValue>) -> Result<RandomForestRegressorModel, JsError> {
        let o = Options::new(options)?;
        Ok(RandomForestRegressorModel {
            n_estimators: o.usize_or("nEstimators", 100)?,
            max_depth: o.usize_or("maxDepth", 10)?,
            min_samples_leaf: o.usize_or("minSamplesLeaf", 1)?,
            max_features: o.usize_opt("maxFeatures")?,
            seed: o.seed_or("seed", 0)?,
            fitted: None,
        })
    }

    /// Fits to a flat row-major matrix and a target with one value per row.
    pub fn fit(&mut self, data: &[f64], cols: usize, y: &[f64]) -> Result<(), JsError> {
        let v = view(data, cols)?;
        check_targets(y, v.rows())?;
        let mut m = RandomForestRegressor::new(self.n_estimators, self.max_depth)
            .with_min_samples_leaf(self.min_samples_leaf)
            .with_seed(self.seed);
        if let Some(f) = self.max_features {
            m = m.with_max_features(f);
        }
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
}

/// A random forest of classification trees, with a majority vote.
///
/// Options: `nEstimators` (100), `maxDepth` (10), `minSamplesLeaf` (1), `criterion` (`"gini"`
/// or `"entropy"`), `maxFeatures` (the square root of the feature count), `seed` (0).
#[wasm_bindgen]
pub struct RandomForestClassifierModel {
    n_estimators: usize,
    max_depth: usize,
    min_samples_leaf: usize,
    criterion: Criterion,
    max_features: Option<usize>,
    seed: u64,
    fitted: Option<RandomForestClassifier>,
}

#[wasm_bindgen]
impl RandomForestClassifierModel {
    #[wasm_bindgen(constructor)]
    pub fn new(options: Option<JsValue>) -> Result<RandomForestClassifierModel, JsError> {
        let o = Options::new(options)?;
        Ok(RandomForestClassifierModel {
            n_estimators: o.usize_or("nEstimators", 100)?,
            max_depth: o.usize_or("maxDepth", 10)?,
            min_samples_leaf: o.usize_or("minSamplesLeaf", 1)?,
            criterion: parse_criterion(&o.str_or("criterion", "gini")?)?,
            max_features: o.usize_opt("maxFeatures")?,
            seed: o.seed_or("seed", 0)?,
            fitted: None,
        })
    }

    /// Fits to a flat row-major matrix and class labels.
    pub fn fit(&mut self, data: &[f64], cols: usize, y: &[f64]) -> Result<(), JsError> {
        let v = view(data, cols)?;
        check_targets(y, v.rows())?;
        let mut m = RandomForestClassifier::new(self.n_estimators, self.max_depth)
            .with_min_samples_leaf(self.min_samples_leaf)
            .with_criterion(self.criterion)
            .with_seed(self.seed);
        if let Some(f) = self.max_features {
            m = m.with_max_features(f);
        }
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

    /// The classes seen in training, sorted.
    #[wasm_bindgen(getter)]
    pub fn classes(&self) -> Result<Vec<f64>, JsError> {
        Ok(self
            .fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .classes()
            .unwrap_or(&[])
            .to_vec())
    }
}
