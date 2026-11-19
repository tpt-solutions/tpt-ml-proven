// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! WebAssembly wrappers for the Phase 8 estimators, in `f64`.
//!
//! Inputs are flat row-major `Float64Array`s with `cols` features per row. Labels come back
//! as `Int32Array` (`-1` for noise) or `Uint32Array` (indices). Matrices come back flat, with
//! the shape given by a `*Rows` or `*Cols` getter where it is not obvious.

use wasm_bindgen::prelude::*;

use tpt_ml_proven_clustering::{Dbscan, Hdbscan};
use tpt_ml_proven_core::{
    ArrayView, DenseView, Predictor, SupervisedEstimator, Transformer, UnsupervisedEstimator,
};
use tpt_ml_proven_decomposition::{Nmf, TruncatedSvd};
use tpt_ml_proven_ensemble::StumpForest;
use tpt_ml_proven_linear::{ElasticNet, Lasso, LogisticRegression, MulticlassLogistic, Ridge};

use crate::{check_shape, flatten, js_err};

fn not_fitted() -> JsError {
    JsError::new("this estimator is not fitted yet; call fit first")
}

fn view(data: &[f64], cols: usize) -> Result<DenseView<'_, f64>, JsError> {
    let rows = check_shape(data.len(), cols)?;
    DenseView::new(data, rows, cols).map_err(|e| js_err(e.into()))
}

fn check_targets(y: &[f64], rows: usize) -> Result<(), JsError> {
    if y.len() != rows {
        return Err(JsError::new(
            "the target has a different length from the number of rows of X",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------- DBSCAN

/// Density-based clustering. Noise samples get label `-1`.
#[wasm_bindgen]
pub struct DbscanModel {
    eps: f64,
    min_samples: usize,
    fitted: Option<Dbscan<f64>>,
}

#[wasm_bindgen]
impl DbscanModel {
    #[wasm_bindgen(constructor)]
    pub fn new(eps: f64, min_samples: usize) -> DbscanModel {
        DbscanModel {
            eps,
            min_samples,
            fitted: None,
        }
    }

    /// Fits to a flat row-major matrix with `cols` features per row.
    pub fn fit(&mut self, data: &[f64], cols: usize) -> Result<(), JsError> {
        let mut m = Dbscan::<f64>::new(self.eps, self.min_samples);
        UnsupervisedEstimator::fit(&mut m, view(data, cols)?).map_err(js_err)?;
        self.fitted = Some(m);
        Ok(())
    }

    /// Cluster label of each training row (`-1` for noise).
    #[wasm_bindgen(getter, js_name = labels)]
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

// ---------------------------------------------------------------- HDBSCAN

/// Hierarchical density-based clustering with excess-of-mass selection.
#[wasm_bindgen]
pub struct HdbscanModel {
    min_cluster_size: usize,
    min_samples: Option<usize>,
    fitted: Option<Hdbscan<f64>>,
}

#[wasm_bindgen]
impl HdbscanModel {
    /// `min_samples` defaults to `min_cluster_size` when omitted.
    #[wasm_bindgen(constructor)]
    pub fn new(min_cluster_size: usize, min_samples: Option<usize>) -> HdbscanModel {
        HdbscanModel {
            min_cluster_size,
            min_samples,
            fitted: None,
        }
    }

    /// Fits to a flat row-major matrix with `cols` features per row.
    pub fn fit(&mut self, data: &[f64], cols: usize) -> Result<(), JsError> {
        let mut m = Hdbscan::<f64>::new(self.min_cluster_size);
        if let Some(ms) = self.min_samples {
            m = m.with_min_samples(ms);
        }
        UnsupervisedEstimator::fit(&mut m, view(data, cols)?).map_err(js_err)?;
        self.fitted = Some(m);
        Ok(())
    }

    /// Cluster label of each training row (`-1` for noise).
    #[wasm_bindgen(getter, js_name = labels)]
    pub fn labels(&self) -> Result<Vec<i32>, JsError> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(m.labels()
            .unwrap_or(&[])
            .iter()
            .map(|&l| l as i32)
            .collect())
    }
}

// ---------------------------------------------------------------- TruncatedSVD

/// Truncated singular value decomposition (dense input, no centring).
#[wasm_bindgen]
pub struct TruncatedSvdModel {
    n_components: usize,
    fitted: Option<TruncatedSvd<f64>>,
}

#[wasm_bindgen]
impl TruncatedSvdModel {
    #[wasm_bindgen(constructor)]
    pub fn new(n_components: usize) -> TruncatedSvdModel {
        TruncatedSvdModel {
            n_components,
            fitted: None,
        }
    }

    /// Fits to a flat row-major matrix with `cols` features per row.
    pub fn fit(&mut self, data: &[f64], cols: usize) -> Result<(), JsError> {
        let mut m = TruncatedSvd::<f64>::new(self.n_components);
        UnsupervisedEstimator::fit(&mut m, view(data, cols)?).map_err(js_err)?;
        self.fitted = Some(m);
        Ok(())
    }

    /// Projection onto the components, flat with `n_components` columns per row.
    pub fn transform(&self, data: &[f64], cols: usize) -> Result<Vec<f64>, JsError> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let out = m.transform(view(data, cols)?).map_err(js_err)?;
        Ok(flatten(&out))
    }

    /// Singular values, largest first.
    #[wasm_bindgen(getter, js_name = singularValues)]
    pub fn singular_values(&self) -> Result<Vec<f64>, JsError> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(m.singular_values().unwrap_or(&[]).to_vec())
    }
}

// ---------------------------------------------------------------- NMF

/// Non-negative matrix factorisation with Frobenius loss and multiplicative updates.
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
    pub fn new(
        n_components: usize,
        max_iter: Option<usize>,
        tol: Option<f64>,
        seed: Option<u64>,
    ) -> NmfModel {
        NmfModel {
            n_components,
            max_iter: max_iter.unwrap_or(200),
            tol: tol.unwrap_or(1e-4),
            seed: seed.unwrap_or(0),
            fitted: None,
        }
    }

    /// Fits to a non-negative flat row-major matrix with `cols` features per row.
    pub fn fit(&mut self, data: &[f64], cols: usize) -> Result<(), JsError> {
        let mut m = Nmf::<f64>::new(self.n_components)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_seed(self.seed);
        UnsupervisedEstimator::fit(&mut m, view(data, cols)?).map_err(js_err)?;
        self.fitted = Some(m);
        Ok(())
    }

    /// Non-negative factor `W` for the rows, flat with `n_components` columns per row.
    pub fn transform(&self, data: &[f64], cols: usize) -> Result<Vec<f64>, JsError> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let out = m.transform(view(data, cols)?).map_err(js_err)?;
        Ok(flatten(&out))
    }

    /// Learned factors `H`, flat with one row per component.
    #[wasm_bindgen(getter)]
    pub fn components(&self) -> Result<Vec<f64>, JsError> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(m.components().unwrap_or(&[]).to_vec())
    }

    /// Multiplicative-update iterations used.
    #[wasm_bindgen(getter, js_name = nIter)]
    pub fn n_iter(&self) -> Result<usize, JsError> {
        self.fitted
            .as_ref()
            .and_then(|m| m.n_iter())
            .ok_or_else(not_fitted)
    }
}

// ---------------------------------------------------------------- linear models

/// Shared shape of the regression wrappers: fit on `(X, y)`, predict, expose coefficients.
macro_rules! regression_class {
    ($name:ident, $rust_ty:ident, $($field:ident : $ty:ty),* ; $build:expr ; $ctor:item) => {
        /// Linear regression, fitted in `f64`.
        #[wasm_bindgen]
        pub struct $name {
            $($field: $ty,)*
            fitted: Option<$rust_ty<f64>>,
        }

        #[wasm_bindgen]
        impl $name {
            $ctor

            /// Fits to a flat row-major matrix with `cols` features per row, and a target.
            pub fn fit(&mut self, data: &[f64], cols: usize, y: &[f64]) -> Result<(), JsError> {
                let v = view(data, cols)?;
                check_targets(y, v.rows())?;
                let build: fn(&Self) -> $rust_ty<f64> = $build;
                let mut model = build(self);
                SupervisedEstimator::fit(&mut model, v, y).map_err(js_err)?;
                self.fitted = Some(model);
                Ok(())
            }

            /// Predicted target for each row.
            pub fn predict(&self, data: &[f64], cols: usize) -> Result<Vec<f64>, JsError> {
                let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
                Predictor::predict(m, view(data, cols)?).map_err(js_err)
            }

            /// Fitted coefficients, one per feature.
            #[wasm_bindgen(getter)]
            pub fn coef(&self) -> Result<Vec<f64>, JsError> {
                let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
                Ok(m.coef().unwrap_or(&[]).to_vec())
            }

            /// Fitted intercept.
            #[wasm_bindgen(getter)]
            pub fn intercept(&self) -> Result<f64, JsError> {
                let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
                m.intercept().ok_or_else(not_fitted)
            }
        }
    };
}

regression_class!(
    RidgeModel, Ridge,
    alpha: f64 ;
    |s: &RidgeModel| Ridge::<f64>::new(s.alpha) ;
    #[wasm_bindgen(constructor)]
    pub fn new(alpha: Option<f64>) -> RidgeModel {
        RidgeModel { alpha: alpha.unwrap_or(1.0), fitted: None }
    }
);

regression_class!(
    LassoModel, Lasso,
    alpha: f64, max_iter: usize, tol: f64 ;
    |s: &LassoModel| Lasso::<f64>::new(s.alpha).with_max_iter(s.max_iter).with_tol(s.tol) ;
    #[wasm_bindgen(constructor)]
    pub fn new(alpha: Option<f64>, max_iter: Option<usize>, tol: Option<f64>) -> LassoModel {
        LassoModel {
            alpha: alpha.unwrap_or(1.0),
            max_iter: max_iter.unwrap_or(1000),
            tol: tol.unwrap_or(1e-4),
            fitted: None,
        }
    }
);

regression_class!(
    ElasticNetModel, ElasticNet,
    alpha: f64, l1_ratio: f64, max_iter: usize, tol: f64 ;
    |s: &ElasticNetModel| ElasticNet::<f64>::new(s.alpha, s.l1_ratio)
        .with_max_iter(s.max_iter)
        .with_tol(s.tol) ;
    #[wasm_bindgen(constructor)]
    pub fn new(alpha: Option<f64>, l1_ratio: Option<f64>, max_iter: Option<usize>, tol: Option<f64>) -> ElasticNetModel {
        ElasticNetModel {
            alpha: alpha.unwrap_or(1.0),
            l1_ratio: l1_ratio.unwrap_or(0.5),
            max_iter: max_iter.unwrap_or(1000),
            tol: tol.unwrap_or(1e-4),
            fitted: None,
        }
    }
);

// ---------------------------------------------------------------- logistic regression

enum LogitState {
    Binary {
        model: LogisticRegression<f64>,
        classes: [f64; 2],
    },
    Multi(MulticlassLogistic<f64>),
}

/// Logistic regression for two or more classes. `C` is the inverse regularisation strength.
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
    pub fn new(
        c: Option<f64>,
        max_iter: Option<usize>,
        tol: Option<f64>,
    ) -> LogisticRegressionModel {
        LogisticRegressionModel {
            c: c.unwrap_or(1.0),
            max_iter: max_iter.unwrap_or(100),
            tol: tol.unwrap_or(1e-4),
            fitted: None,
        }
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

// ---------------------------------------------------------------- stump forest

/// Bagged depth-one classification trees with a majority vote.
#[wasm_bindgen]
pub struct StumpForestModel {
    n_estimators: usize,
    seed: u64,
    fitted: Option<StumpForest<f64>>,
}

#[wasm_bindgen]
impl StumpForestModel {
    #[wasm_bindgen(constructor)]
    pub fn new(n_estimators: Option<usize>, seed: Option<u64>) -> StumpForestModel {
        StumpForestModel {
            n_estimators: n_estimators.unwrap_or(100),
            seed: seed.unwrap_or(0),
            fitted: None,
        }
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
}
