// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Python wrappers for the Phase 8 estimators. These run in `f64`: the input is copied
//! into `f64` before fitting, so a `float32` array gives the same result as its `float64`
//! copy. The copy is made while the GIL is held, and fitting runs without it.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use tpt_ml_proven_clustering::{Dbscan, Hdbscan};
use tpt_ml_proven_core::error::Error;
use tpt_ml_proven_core::{
    DenseView, Predictor, SupervisedEstimator, Transformer, UnsupervisedEstimator,
};
use tpt_ml_proven_decomposition::{Nmf, TruncatedSvd};
use tpt_ml_proven_ensemble::StumpForest;
use tpt_ml_proven_linear::{ElasticNet, Lasso, LogisticRegression, MulticlassLogistic, Ridge};

use crate::{matrix_from, not_fitted, rows_of, to_py_err, to_vec};

/// Copies a 2-D array into a row-major `f64` buffer, with its shape.
fn read_matrix(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, usize, usize)> {
    let m = matrix_from(py, x)?;
    let (r, c) = m.shape();
    Ok((to_vec::<f64>(&m), r, c))
}

/// Copies a 1-D target sequence into `f64`.
fn read_targets(y: Vec<f64>, rows: usize) -> PyResult<Vec<f64>> {
    if y.len() != rows {
        return Err(PyValueError::new_err(
            "the target has a different length from the number of rows of X",
        ));
    }
    Ok(y)
}

// ---------------------------------------------------------------- DBSCAN

/// Density-based clustering. Noise samples get label `-1`.
#[pyclass(name = "DBSCAN", module = "tpt_ml_proven")]
pub struct PyDbscan {
    eps: f64,
    min_samples: usize,
    fitted: Option<Dbscan<f64>>,
}

#[pymethods]
impl PyDbscan {
    #[new]
    fn new(eps: f64, min_samples: usize) -> Self {
        Self {
            eps,
            min_samples,
            fitted: None,
        }
    }

    /// Fits to a 2-D array and returns the estimator. Releases the GIL while fitting.
    fn fit<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let (data, r, c) = read_matrix(py, x)?;
        let (eps, min_samples) = {
            let b = slf.borrow();
            (b.eps, b.min_samples)
        };
        let fitted = py.detach(move || -> Result<Dbscan<f64>, Error> {
            let mut m = Dbscan::<f64>::new(eps, min_samples);
            UnsupervisedEstimator::fit(&mut m, DenseView::new(&data, r, c)?)?;
            Ok(m)
        });
        slf.borrow_mut().fitted = Some(fitted.map_err(to_py_err)?);
        Ok(slf)
    }

    /// Cluster label of each training row, `-1` for noise.
    #[getter]
    fn labels_(&self) -> PyResult<Vec<i64>> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(m.labels().unwrap_or(&[]).to_vec())
    }

    /// Indices of the core samples.
    #[getter]
    fn core_sample_indices_(&self) -> PyResult<Vec<usize>> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(m.core_sample_indices().unwrap_or(&[]).to_vec())
    }
}

// ---------------------------------------------------------------- HDBSCAN

/// Hierarchical density-based clustering with excess-of-mass selection.
#[pyclass(name = "HDBSCAN", module = "tpt_ml_proven")]
pub struct PyHdbscan {
    min_cluster_size: usize,
    min_samples: Option<usize>,
    fitted: Option<Hdbscan<f64>>,
}

#[pymethods]
impl PyHdbscan {
    #[new]
    #[pyo3(signature = (min_cluster_size=5, *, min_samples=None))]
    fn new(min_cluster_size: usize, min_samples: Option<usize>) -> Self {
        Self {
            min_cluster_size,
            min_samples,
            fitted: None,
        }
    }

    /// Fits to a 2-D array and returns the estimator. Releases the GIL while fitting.
    fn fit<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let (data, r, c) = read_matrix(py, x)?;
        let (mcs, ms) = {
            let b = slf.borrow();
            (b.min_cluster_size, b.min_samples)
        };
        let fitted = py.detach(move || -> Result<Hdbscan<f64>, Error> {
            let mut m = Hdbscan::<f64>::new(mcs);
            if let Some(ms) = ms {
                m = m.with_min_samples(ms);
            }
            UnsupervisedEstimator::fit(&mut m, DenseView::new(&data, r, c)?)?;
            Ok(m)
        });
        slf.borrow_mut().fitted = Some(fitted.map_err(to_py_err)?);
        Ok(slf)
    }

    /// Cluster label of each training row, `-1` for noise.
    #[getter]
    fn labels_(&self) -> PyResult<Vec<i64>> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(m.labels().unwrap_or(&[]).to_vec())
    }
}

// ---------------------------------------------------------------- TruncatedSVD

/// Truncated singular value decomposition (dense input, no centring).
#[pyclass(name = "TruncatedSVD", module = "tpt_ml_proven")]
pub struct PyTruncatedSvd {
    n_components: usize,
    fitted: Option<TruncatedSvd<f64>>,
}

#[pymethods]
impl PyTruncatedSvd {
    #[new]
    fn new(n_components: usize) -> Self {
        Self {
            n_components,
            fitted: None,
        }
    }

    /// Fits to a 2-D array and returns the estimator. Releases the GIL while fitting.
    fn fit<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let (data, r, c) = read_matrix(py, x)?;
        let k = slf.borrow().n_components;
        let fitted = py.detach(move || -> Result<TruncatedSvd<f64>, Error> {
            let mut m = TruncatedSvd::<f64>::new(k);
            UnsupervisedEstimator::fit(&mut m, DenseView::new(&data, r, c)?)?;
            Ok(m)
        });
        slf.borrow_mut().fitted = Some(fitted.map_err(to_py_err)?);
        Ok(slf)
    }

    /// Projection of the rows onto the components.
    fn transform(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let (data, r, c) = read_matrix(py, x)?;
        let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
        Ok(rows_of(&m.transform(view).map_err(to_py_err)?))
    }

    /// Right singular vectors, one per row.
    #[getter]
    fn components_(&self) -> PyResult<Vec<Vec<f64>>> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(m.components().unwrap_or_default())
    }

    /// Singular values, largest first.
    #[getter]
    fn singular_values_(&self) -> PyResult<Vec<f64>> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(m.singular_values().unwrap_or(&[]).to_vec())
    }
}

// ---------------------------------------------------------------- NMF

/// Non-negative matrix factorisation with Frobenius loss and multiplicative updates.
#[pyclass(name = "NMF", module = "tpt_ml_proven")]
pub struct PyNmf {
    n_components: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    fitted: Option<Nmf<f64>>,
}

#[pymethods]
impl PyNmf {
    #[new]
    #[pyo3(signature = (n_components, *, max_iter=200, tol=1e-4, seed=0))]
    fn new(n_components: usize, max_iter: usize, tol: f64, seed: u64) -> Self {
        Self {
            n_components,
            max_iter,
            tol,
            seed,
            fitted: None,
        }
    }

    /// Fits to a non-negative 2-D array and returns the estimator. Releases the GIL.
    fn fit<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let (data, r, c) = read_matrix(py, x)?;
        let (k, max_iter, tol, seed) = {
            let b = slf.borrow();
            (b.n_components, b.max_iter, b.tol, b.seed)
        };
        let fitted = py.detach(move || -> Result<Nmf<f64>, Error> {
            let mut m = Nmf::<f64>::new(k)
                .with_max_iter(max_iter)
                .with_tol(tol)
                .with_seed(seed);
            UnsupervisedEstimator::fit(&mut m, DenseView::new(&data, r, c)?)?;
            Ok(m)
        });
        slf.borrow_mut().fitted = Some(fitted.map_err(to_py_err)?);
        Ok(slf)
    }

    /// Non-negative factor `W` for the rows, with the fitted components held fixed.
    fn transform(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let (data, r, c) = read_matrix(py, x)?;
        let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
        Ok(rows_of(&m.transform(view).map_err(to_py_err)?))
    }

    /// Factors `H` (`n_components` rows), learned from the training data.
    #[getter]
    fn components_(&self) -> PyResult<Vec<Vec<f64>>> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let h = m.components().unwrap_or(&[]);
        let cols = if self.n_components == 0 {
            0
        } else {
            h.len() / self.n_components
        };
        Ok(h.chunks(cols.max(1)).map(<[f64]>::to_vec).collect())
    }

    /// Multiplicative-update iterations used.
    #[getter]
    fn n_iter_(&self) -> PyResult<usize> {
        self.fitted
            .as_ref()
            .and_then(|m| m.n_iter())
            .ok_or_else(not_fitted)
    }

    /// Frobenius norm of `X − WH` at the end of the fit.
    #[getter]
    fn reconstruction_err_(&self) -> PyResult<f64> {
        self.fitted
            .as_ref()
            .and_then(|m| m.reconstruction_error())
            .ok_or_else(not_fitted)
    }
}

// ---------------------------------------------------------------- linear models

/// Shared shape of the regression wrappers: fit on `(X, y)`, predict, expose coefficients.
macro_rules! regression_class {
    ($py_name:literal, $rust_ty:ident, $ident:ident, $($field:ident : $ty:ty),* ; $build:expr ; $ctor:item) => {
        #[doc = concat!($py_name, " regression, fitted in f64.")]
        #[pyclass(name = $py_name, module = "tpt_ml_proven")]
        pub struct $ident {
            $($field: $ty,)*
            fitted: Option<$rust_ty<f64>>,
        }

        #[pymethods]
        impl $ident {
            $ctor

            /// Fits to a 2-D array and a target sequence. Releases the GIL while fitting.
            fn fit<'py>(
                slf: Bound<'py, Self>,
                py: Python<'py>,
                x: &Bound<'py, PyAny>,
                y: Vec<f64>,
            ) -> PyResult<Bound<'py, Self>> {
                let (data, r, c) = read_matrix(py, x)?;
                let y = read_targets(y, r)?;
                let model = {
                    let b = slf.borrow();
                    let build: fn(&Self) -> $rust_ty<f64> = $build;
                    build(&b)
                };
                let mut model = model;
                let fitted = py.detach(move || -> Result<$rust_ty<f64>, Error> {
                    SupervisedEstimator::fit(&mut model, DenseView::new(&data, r, c)?, &y[..])?;
                    Ok(model)
                });
                slf.borrow_mut().fitted = Some(fitted.map_err(to_py_err)?);
                Ok(slf)
            }

            /// Predicted target for each row.
            fn predict(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
                let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
                let (data, r, c) = read_matrix(py, x)?;
                let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
                Predictor::predict(m, view).map_err(to_py_err)
            }

            /// Fitted coefficients, one per feature.
            #[getter]
            fn coef_(&self) -> PyResult<Vec<f64>> {
                let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
                Ok(m.coef().unwrap_or(&[]).to_vec())
            }

            /// Fitted intercept.
            #[getter]
            fn intercept_(&self) -> PyResult<f64> {
                let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
                m.intercept().ok_or_else(not_fitted)
            }
        }
    };
}

regression_class!(
    "Ridge", Ridge, PyRidge,
    alpha: f64 ;
    |s: &PyRidge| Ridge::<f64>::new(s.alpha) ;
    #[new]
    #[pyo3(signature = (alpha=1.0))]
    fn new(alpha: f64) -> Self {
        Self { alpha, fitted: None }
    }
);

regression_class!(
    "Lasso", Lasso, PyLasso,
    alpha: f64, max_iter: usize, tol: f64 ;
    |s: &PyLasso| Lasso::<f64>::new(s.alpha).with_max_iter(s.max_iter).with_tol(s.tol) ;
    #[new]
    #[pyo3(signature = (alpha=1.0, *, max_iter=1000, tol=1e-4))]
    fn new(alpha: f64, max_iter: usize, tol: f64) -> Self {
        Self { alpha, max_iter, tol, fitted: None }
    }
);

regression_class!(
    "ElasticNet", ElasticNet, PyElasticNet,
    alpha: f64, l1_ratio: f64, max_iter: usize, tol: f64 ;
    |s: &PyElasticNet| ElasticNet::<f64>::new(s.alpha, s.l1_ratio)
        .with_max_iter(s.max_iter)
        .with_tol(s.tol) ;
    #[new]
    #[pyo3(signature = (alpha=1.0, l1_ratio=0.5, *, max_iter=1000, tol=1e-4))]
    fn new(alpha: f64, l1_ratio: f64, max_iter: usize, tol: f64) -> Self {
        Self { alpha, l1_ratio, max_iter, tol, fitted: None }
    }
);

// ---------------------------------------------------------------- logistic regression

enum LogitState {
    /// Two classes, mapped to 0 and 1.
    Binary {
        model: LogisticRegression<f64>,
        classes: [f64; 2],
    },
    /// Three or more classes.
    Multi(MulticlassLogistic<f64>),
}

/// Logistic regression for two or more classes (`C` is the inverse regularisation
/// strength, as in scikit-learn). With two classes it is binary; otherwise multinomial.
#[pyclass(name = "LogisticRegression", module = "tpt_ml_proven")]
pub struct PyLogisticRegression {
    c: f64,
    max_iter: usize,
    tol: f64,
    fitted: Option<LogitState>,
}

#[pymethods]
impl PyLogisticRegression {
    #[new]
    #[allow(non_snake_case)]
    #[pyo3(signature = (C=1.0, *, max_iter=100, tol=1e-4))]
    fn new(C: f64, max_iter: usize, tol: f64) -> Self {
        Self {
            c: C,
            max_iter,
            tol,
            fitted: None,
        }
    }

    /// Fits to a 2-D array and class labels. Releases the GIL while fitting.
    fn fit<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: Vec<f64>,
    ) -> PyResult<Bound<'py, Self>> {
        let (data, r, c) = read_matrix(py, x)?;
        let y = read_targets(y, r)?;
        let (cc, max_iter, tol) = {
            let b = slf.borrow();
            (b.c, b.max_iter, b.tol)
        };
        let mut classes = y.clone();
        classes.sort_by(f64::total_cmp);
        classes.dedup();

        let state = if classes.len() == 2 {
            let y01: Vec<f64> = y
                .iter()
                .map(|&v| if v == classes[1] { 1.0 } else { 0.0 })
                .collect();
            let fitted = py.detach(move || -> Result<LogisticRegression<f64>, Error> {
                let mut m = LogisticRegression::<f64>::new(cc)
                    .with_max_iter(max_iter)
                    .with_tol(tol);
                SupervisedEstimator::fit(&mut m, DenseView::new(&data, r, c)?, &y01[..])?;
                Ok(m)
            });
            LogitState::Binary {
                model: fitted.map_err(to_py_err)?,
                classes: [classes[0], classes[1]],
            }
        } else {
            let fitted = py.detach(move || -> Result<MulticlassLogistic<f64>, Error> {
                let mut m = MulticlassLogistic::<f64>::new(cc)
                    .with_max_iter(max_iter)
                    .with_tol(tol);
                SupervisedEstimator::fit(&mut m, DenseView::new(&data, r, c)?, &y[..])?;
                Ok(m)
            });
            LogitState::Multi(fitted.map_err(to_py_err)?)
        };
        slf.borrow_mut().fitted = Some(state);
        Ok(slf)
    }

    /// Predicted class label for each row.
    fn predict(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let state = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let (data, r, c) = read_matrix(py, x)?;
        let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
        match state {
            LogitState::Binary { model, classes } => {
                let zero_one = Predictor::predict(model, view).map_err(to_py_err)?;
                Ok(zero_one
                    .into_iter()
                    .map(|v| if v == 1.0 { classes[1] } else { classes[0] })
                    .collect())
            }
            LogitState::Multi(model) => Predictor::predict(model, view).map_err(to_py_err),
        }
    }

    /// Class probabilities, one row per sample, columns in the order of `classes_`.
    fn predict_proba(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        let state = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let (data, r, c) = read_matrix(py, x)?;
        let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
        match state {
            LogitState::Binary { model, .. } => {
                let p = model.predict_proba(view).map_err(to_py_err)?;
                Ok(p.into_iter().map(|p1| vec![1.0 - p1, p1]).collect())
            }
            LogitState::Multi(model) => {
                let k = model.classes().map_or(0, <[f64]>::len).max(1);
                let flat = model.predict_proba(view).map_err(to_py_err)?;
                Ok(flat.chunks(k).map(|row| row.to_vec()).collect())
            }
        }
    }

    /// The classes seen in training, sorted.
    #[getter]
    fn classes_(&self) -> PyResult<Vec<f64>> {
        match self.fitted.as_ref().ok_or_else(not_fitted)? {
            LogitState::Binary { classes, .. } => Ok(classes.to_vec()),
            LogitState::Multi(model) => Ok(model.classes().unwrap_or(&[]).to_vec()),
        }
    }
}

// ---------------------------------------------------------------- stump forest

/// Bagged depth-one classification trees with a majority vote.
#[pyclass(name = "StumpForest", module = "tpt_ml_proven")]
pub struct PyStumpForest {
    n_estimators: usize,
    seed: u64,
    fitted: Option<StumpForest<f64>>,
}

#[pymethods]
impl PyStumpForest {
    #[new]
    #[pyo3(signature = (n_estimators=100, *, seed=0))]
    fn new(n_estimators: usize, seed: u64) -> Self {
        Self {
            n_estimators,
            seed,
            fitted: None,
        }
    }

    /// Fits to a 2-D array and class labels. Releases the GIL while fitting.
    fn fit<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: Vec<f64>,
    ) -> PyResult<Bound<'py, Self>> {
        let (data, r, c) = read_matrix(py, x)?;
        let y = read_targets(y, r)?;
        let (k, seed) = {
            let b = slf.borrow();
            (b.n_estimators, b.seed)
        };
        let fitted = py.detach(move || -> Result<StumpForest<f64>, Error> {
            let mut m = StumpForest::<f64>::new(k).with_seed(seed);
            SupervisedEstimator::fit(&mut m, DenseView::new(&data, r, c)?, &y[..])?;
            Ok(m)
        });
        slf.borrow_mut().fitted = Some(fitted.map_err(to_py_err)?);
        Ok(slf)
    }

    /// Predicted class label for each row.
    fn predict(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let (data, r, c) = read_matrix(py, x)?;
        let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
        Predictor::predict(m, view).map_err(to_py_err)
    }
}
