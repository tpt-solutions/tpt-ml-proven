// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Python wrappers for the Phase 8 estimators. These run in `f64`: the input is copied
//! into `f64` before fitting, so a `float32` array gives the same result as its `float64`
//! copy. The copy is made while the GIL is held, and fitting runs without it.

use crate::{repr_of, sklearn_tags};
use pyo3::exceptions::PyAttributeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};

use tpt_ml_proven_clustering::{Dbscan, Hdbscan};
use tpt_ml_proven_core::error::Error;
use tpt_ml_proven_core::persist::kind;
use tpt_ml_proven_core::{
    ArrayView, DenseView, Predictor, SupervisedEstimator, Transformer, UnsupervisedEstimator,
};
use tpt_ml_proven_decomposition::{Nmf, TruncatedSvd};
use tpt_ml_proven_ensemble::StumpForest;
use tpt_ml_proven_linear::{
    labelled_binary_from_bytes, labelled_binary_to_bytes, ElasticNet, Lasso, LogisticRegression,
    MulticlassLogistic, Ridge,
};

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
#[pyclass(name = "DBSCAN", module = "tpt_ml_proven", dict)]
pub struct PyDbscan {
    eps: f64,
    min_samples: usize,
    fitted: Option<Dbscan<f64>>,
}

impl PyDbscan {
    fn encode(&self) -> PyResult<Vec<u8>> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .to_bytes()
            .map_err(to_py_err)
    }

    fn decode(bytes: &[u8]) -> PyResult<Self> {
        let m = Dbscan::<f64>::from_bytes(bytes).map_err(to_py_err)?;
        Ok(Self {
            eps: m.eps(),
            min_samples: m.min_samples(),
            fitted: Some(m),
        })
    }
}

#[pymethods]
impl PyDbscan {
    /// Serialises the fitted model to bytes, in the tpt-ml-proven model file format.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &self.encode()?))
    }

    /// Loads a model from bytes written by `to_bytes`. The hyper-parameters come from the file.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        Self::decode(data)
    }

    /// Writes the fitted model to a file.
    fn save(&self, path: &str) -> PyResult<()> {
        std::fs::write(path, self.encode()?).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Loads a model from a file written by `save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let bytes = std::fs::read(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Self::decode(&bytes)
    }

    /// Pickle support: unpickling calls `from_bytes` on the saved bytes.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let py = slf.py();
        let bytes = PyBytes::new(py, &slf.borrow().encode()?);
        let loader = slf.get_type().getattr("from_bytes")?;
        Ok((loader, (bytes,)))
    }

    /// Fits, then returns the cluster label of each row (`-1` for noise).
    fn fit_predict<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<i64>> {
        let fitted = Self::fit(slf, py, x)?;
        let labels = fitted.borrow().labels_()?;
        Ok(labels)
    }

    /// scikit-learn's fitted check: true once `fit` has succeeded.
    fn __sklearn_is_fitted__(&self) -> bool {
        self.fitted.is_some()
    }

    /// scikit-learn's tags protocol: what kind of estimator this is.
    fn __sklearn_tags__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        sklearn_tags(py, "clusterer")
    }

    /// The constructor parameters, by name.
    #[pyo3(signature = (deep=true))]
    #[allow(unused_variables)]
    fn get_params<'py>(&self, py: Python<'py>, deep: bool) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("eps", self.eps)?;
        d.set_item("min_samples", self.min_samples)?;
        Ok(d)
    }

    /// Sets constructor parameters by name and returns the estimator. A fitted model keeps its
    /// fitted state until it is refitted.
    #[pyo3(signature = (**params))]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        params: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        if let Some(params) = params {
            let mut this = slf.borrow_mut();
            for (key, value) in params.iter() {
                let key: String = key.extract()?;
                match key.as_str() {
                    "eps" => this.eps = value.extract()?,
                    "min_samples" => this.min_samples = value.extract()?,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter {other:?} for estimator DBSCAN"
                        )))
                    }
                }
            }
        }
        Ok(slf)
    }

    /// scikit-learn's clone protocol: a new, unfitted estimator with the same parameters.
    fn __sklearn_clone__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let params = slf.borrow().get_params(slf.py(), false)?;
        slf.get_type().call((), Some(&params))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        repr_of("DBSCAN", &self.get_params(py, false)?)
    }

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
#[pyclass(name = "HDBSCAN", module = "tpt_ml_proven", dict)]
pub struct PyHdbscan {
    min_cluster_size: usize,
    min_samples: Option<usize>,
    fitted: Option<Hdbscan<f64>>,
}

impl PyHdbscan {
    fn encode(&self) -> PyResult<Vec<u8>> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .to_bytes()
            .map_err(to_py_err)
    }

    fn decode(bytes: &[u8]) -> PyResult<Self> {
        let m = Hdbscan::<f64>::from_bytes(bytes).map_err(to_py_err)?;
        Ok(Self {
            min_cluster_size: m.min_cluster_size(),
            min_samples: m.min_samples(),
            fitted: Some(m),
        })
    }
}

#[pymethods]
impl PyHdbscan {
    /// Serialises the fitted model to bytes, in the tpt-ml-proven model file format.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &self.encode()?))
    }

    /// Loads a model from bytes written by `to_bytes`. The hyper-parameters come from the file.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        Self::decode(data)
    }

    /// Writes the fitted model to a file.
    fn save(&self, path: &str) -> PyResult<()> {
        std::fs::write(path, self.encode()?).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Loads a model from a file written by `save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let bytes = std::fs::read(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Self::decode(&bytes)
    }

    /// Pickle support: unpickling calls `from_bytes` on the saved bytes.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let py = slf.py();
        let bytes = PyBytes::new(py, &slf.borrow().encode()?);
        let loader = slf.get_type().getattr("from_bytes")?;
        Ok((loader, (bytes,)))
    }

    /// Fits, then returns the cluster label of each row (`-1` for noise).
    fn fit_predict<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<i64>> {
        let fitted = Self::fit(slf, py, x)?;
        let labels = fitted.borrow().labels_()?;
        Ok(labels)
    }

    /// scikit-learn's fitted check: true once `fit` has succeeded.
    fn __sklearn_is_fitted__(&self) -> bool {
        self.fitted.is_some()
    }

    /// scikit-learn's tags protocol: what kind of estimator this is.
    fn __sklearn_tags__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        sklearn_tags(py, "clusterer")
    }

    /// The constructor parameters, by name.
    #[pyo3(signature = (deep=true))]
    #[allow(unused_variables)]
    fn get_params<'py>(&self, py: Python<'py>, deep: bool) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("min_cluster_size", self.min_cluster_size)?;
        d.set_item("min_samples", self.min_samples)?;
        Ok(d)
    }

    /// Sets constructor parameters by name and returns the estimator. A fitted model keeps its
    /// fitted state until it is refitted.
    #[pyo3(signature = (**params))]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        params: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        if let Some(params) = params {
            let mut this = slf.borrow_mut();
            for (key, value) in params.iter() {
                let key: String = key.extract()?;
                match key.as_str() {
                    "min_cluster_size" => this.min_cluster_size = value.extract()?,
                    "min_samples" => this.min_samples = value.extract()?,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter {other:?} for estimator HDBSCAN"
                        )))
                    }
                }
            }
        }
        Ok(slf)
    }

    /// scikit-learn's clone protocol: a new, unfitted estimator with the same parameters.
    fn __sklearn_clone__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let params = slf.borrow().get_params(slf.py(), false)?;
        slf.get_type().call((), Some(&params))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        repr_of("HDBSCAN", &self.get_params(py, false)?)
    }

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
#[pyclass(name = "TruncatedSVD", module = "tpt_ml_proven", dict)]
pub struct PyTruncatedSvd {
    n_components: usize,
    fitted: Option<TruncatedSvd<f64>>,
}

impl PyTruncatedSvd {
    fn encode(&self) -> PyResult<Vec<u8>> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .to_bytes()
            .map_err(to_py_err)
    }

    fn decode(bytes: &[u8]) -> PyResult<Self> {
        let m = TruncatedSvd::<f64>::from_bytes(bytes).map_err(to_py_err)?;
        Ok(Self {
            n_components: m.n_components(),
            fitted: Some(m),
        })
    }
}

#[pymethods]
impl PyTruncatedSvd {
    /// Number of features seen in `fit`. Raises AttributeError before fit, as scikit-learn does.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        (self
            .fitted
            .as_ref()
            .and_then(|m| m.components().map(|d| d.cols())))
        .ok_or_else(|| PyAttributeError::new_err("n_features_in_ is not available before fit"))
    }

    /// scikit-learn's fitted check: true once `fit` has succeeded.
    fn __sklearn_is_fitted__(&self) -> bool {
        self.fitted.is_some()
    }

    /// scikit-learn's tags protocol: what kind of estimator this is.
    fn __sklearn_tags__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        sklearn_tags(py, "transformer")
    }

    /// The constructor parameters, by name.
    #[pyo3(signature = (deep=true))]
    #[allow(unused_variables)]
    fn get_params<'py>(&self, py: Python<'py>, deep: bool) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("n_components", self.n_components)?;
        Ok(d)
    }

    /// Sets constructor parameters by name and returns the estimator. A fitted model keeps its
    /// fitted state until it is refitted.
    #[pyo3(signature = (**params))]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        params: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        if let Some(params) = params {
            let mut this = slf.borrow_mut();
            for (key, value) in params.iter() {
                let key: String = key.extract()?;
                match key.as_str() {
                    "n_components" => this.n_components = value.extract()?,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter {other:?} for estimator TruncatedSVD"
                        )))
                    }
                }
            }
        }
        Ok(slf)
    }

    /// scikit-learn's clone protocol: a new, unfitted estimator with the same parameters.
    fn __sklearn_clone__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let params = slf.borrow().get_params(slf.py(), false)?;
        slf.get_type().call((), Some(&params))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        repr_of("TruncatedSVD", &self.get_params(py, false)?)
    }

    /// Serialises the fitted model to bytes, in the tpt-ml-proven model file format.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &self.encode()?))
    }

    /// Loads a model from bytes written by `to_bytes`. The hyper-parameters come from the file.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        Self::decode(data)
    }

    /// Writes the fitted model to a file.
    fn save(&self, path: &str) -> PyResult<()> {
        std::fs::write(path, self.encode()?).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Loads a model from a file written by `save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let bytes = std::fs::read(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Self::decode(&bytes)
    }

    /// Pickle support: unpickling calls `from_bytes` on the saved bytes.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let py = slf.py();
        let bytes = PyBytes::new(py, &slf.borrow().encode()?);
        let loader = slf.get_type().getattr("from_bytes")?;
        Ok((loader, (bytes,)))
    }

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
        let d = m.components().ok_or_else(not_fitted)?;
        let cols = d.cols().max(1);
        Ok(d.as_slice().chunks(cols).map(<[f64]>::to_vec).collect())
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
#[pyclass(name = "NMF", module = "tpt_ml_proven", dict)]
pub struct PyNmf {
    n_components: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    fitted: Option<Nmf<f64>>,
}

impl PyNmf {
    fn encode(&self) -> PyResult<Vec<u8>> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .to_bytes()
            .map_err(to_py_err)
    }

    fn decode(bytes: &[u8]) -> PyResult<Self> {
        let m = Nmf::<f64>::from_bytes(bytes).map_err(to_py_err)?;
        Ok(Self {
            n_components: m.n_components(),
            max_iter: m.max_iter(),
            tol: m.tol(),
            seed: m.seed(),
            fitted: Some(m),
        })
    }
}

#[pymethods]
impl PyNmf {
    /// Number of features seen in `fit`. Raises AttributeError before fit, as scikit-learn does.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        (self
            .fitted
            .as_ref()
            .and_then(|m| m.components().map(|d| d.cols())))
        .ok_or_else(|| PyAttributeError::new_err("n_features_in_ is not available before fit"))
    }

    /// scikit-learn's fitted check: true once `fit` has succeeded.
    fn __sklearn_is_fitted__(&self) -> bool {
        self.fitted.is_some()
    }

    /// scikit-learn's tags protocol: what kind of estimator this is.
    fn __sklearn_tags__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        sklearn_tags(py, "transformer")
    }

    /// The constructor parameters, by name.
    #[pyo3(signature = (deep=true))]
    #[allow(unused_variables)]
    fn get_params<'py>(&self, py: Python<'py>, deep: bool) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("n_components", self.n_components)?;
        d.set_item("max_iter", self.max_iter)?;
        d.set_item("tol", self.tol)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Sets constructor parameters by name and returns the estimator. A fitted model keeps its
    /// fitted state until it is refitted.
    #[pyo3(signature = (**params))]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        params: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        if let Some(params) = params {
            let mut this = slf.borrow_mut();
            for (key, value) in params.iter() {
                let key: String = key.extract()?;
                match key.as_str() {
                    "n_components" => this.n_components = value.extract()?,
                    "max_iter" => this.max_iter = value.extract()?,
                    "tol" => this.tol = value.extract()?,
                    "seed" => this.seed = value.extract()?,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter {other:?} for estimator NMF"
                        )))
                    }
                }
            }
        }
        Ok(slf)
    }

    /// scikit-learn's clone protocol: a new, unfitted estimator with the same parameters.
    fn __sklearn_clone__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let params = slf.borrow().get_params(slf.py(), false)?;
        slf.get_type().call((), Some(&params))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        repr_of("NMF", &self.get_params(py, false)?)
    }

    /// Serialises the fitted model to bytes, in the tpt-ml-proven model file format.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &self.encode()?))
    }

    /// Loads a model from bytes written by `to_bytes`. The hyper-parameters come from the file.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        Self::decode(data)
    }

    /// Writes the fitted model to a file.
    fn save(&self, path: &str) -> PyResult<()> {
        std::fs::write(path, self.encode()?).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Loads a model from a file written by `save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let bytes = std::fs::read(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Self::decode(&bytes)
    }

    /// Pickle support: unpickling calls `from_bytes` on the saved bytes.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let py = slf.py();
        let bytes = PyBytes::new(py, &slf.borrow().encode()?);
        let loader = slf.get_type().getattr("from_bytes")?;
        Ok((loader, (bytes,)))
    }

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
        let h = m.components().map_or(&[][..], |d| d.as_slice());
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
    ($py_name:literal, $rust_ty:ident, $ident:ident, $($field:ident : $ty:ty),* ; $build:expr ; $from:expr ; $ctor:item) => {
        #[doc = concat!($py_name, " regression, fitted in f64.")]
        #[pyclass(name = $py_name, module = "tpt_ml_proven", dict)]
        pub struct $ident {
            $($field: $ty,)*
            fitted: Option<$rust_ty<f64>>,
        }

        impl $ident {
            fn encode(&self) -> PyResult<Vec<u8>> {
                self.fitted.as_ref().ok_or_else(not_fitted)?.to_bytes().map_err(to_py_err)
            }

            fn decode(bytes: &[u8]) -> PyResult<Self> {
                let m = <$rust_ty<f64>>::from_bytes(bytes).map_err(to_py_err)?;
                let from: fn($rust_ty<f64>) -> Self = $from;
                Ok(from(m))
            }
        }

        #[pymethods]
        impl $ident {
            $ctor

    /// Number of features seen in `fit`. Raises AttributeError before fit, as scikit-learn does.
        #[getter]
        fn n_features_in_(&self) -> PyResult<usize> {
            (self.fitted.as_ref().and_then(|m| m.coef().map(|c| c.len()))).ok_or_else(|| PyAttributeError::new_err("n_features_in_ is not available before fit"))
        }

            /// scikit-learn's fitted check: true once `fit` has succeeded.
            fn __sklearn_is_fitted__(&self) -> bool {
                self.fitted.is_some()
            }

            /// scikit-learn's tags protocol: a regressor.
            fn __sklearn_tags__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                sklearn_tags(py, "regressor")
            }

            /// The constructor parameters, by name.
            #[pyo3(signature = (deep=true))]
    #[allow(unused_variables)]
    fn get_params<'py>(&self, py: Python<'py>, deep: bool) -> PyResult<Bound<'py, PyDict>> {
                let d = PyDict::new(py);
                $( d.set_item(stringify!($field), self.$field)?; )*
                Ok(d)
            }

            /// Sets constructor parameters by name and returns the estimator. A fitted model keeps
            /// its fitted state until it is refitted.
            #[pyo3(signature = (**params))]
            fn set_params<'py>(
                slf: Bound<'py, Self>,
                params: Option<&Bound<'py, PyDict>>,
            ) -> PyResult<Bound<'py, Self>> {
                if let Some(params) = params {
                    let mut this = slf.borrow_mut();
                    for (key, value) in params.iter() {
                        let key: String = key.extract()?;
                        $(
                            if key == stringify!($field) {
                                this.$field = value.extract()?;
                                continue;
                            }
                        )*
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter {key:?} for estimator {}",
                            $py_name
                        )));
                    }
                }
                Ok(slf)
            }

            /// scikit-learn's clone protocol: a new, unfitted estimator with the same parameters.
    fn __sklearn_clone__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let params = slf.borrow().get_params(slf.py(), false)?;
        slf.get_type().call((), Some(&params))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
                repr_of($py_name, &self.get_params(py, false)?)
            }

            /// Coefficient of determination R² of the predictions for `x` against `y`.
            fn score(&self, py: Python<'_>, x: &Bound<'_, PyAny>, y: Vec<f64>) -> PyResult<f64> {
                let pred = self.predict(py, x)?;
                tpt_ml_proven_core::metrics::r2(&y, &pred).map_err(to_py_err)
            }

                /// Serialises the fitted model to bytes, in the tpt-ml-proven model file format.
                fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
                    Ok(PyBytes::new(py, &self.encode()?))
                }

                /// Loads a model from bytes written by `to_bytes`. The hyper-parameters come from the file.
                #[staticmethod]
                fn from_bytes(data: &[u8]) -> PyResult<Self> {
                    Self::decode(data)
                }

                /// Writes the fitted model to a file.
                fn save(&self, path: &str) -> PyResult<()> {
                    std::fs::write(path, self.encode()?).map_err(|e| PyValueError::new_err(e.to_string()))
                }

                /// Loads a model from a file written by `save`.
                #[staticmethod]
                fn load(path: &str) -> PyResult<Self> {
                    let bytes = std::fs::read(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
                    Self::decode(&bytes)
                }

                /// Pickle support: unpickling calls `from_bytes` on the saved bytes.
                fn __reduce__<'py>(
                    slf: &Bound<'py, Self>,
                ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
                    let py = slf.py();
                    let bytes = PyBytes::new(py, &slf.borrow().encode()?);
                    let loader = slf.get_type().getattr("from_bytes")?;
                    Ok((loader, (bytes,)))
                }




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
    |m: Ridge<f64>| PyRidge { alpha: m.alpha(), fitted: Some(m) } ;
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
    |m: Lasso<f64>| PyLasso { alpha: m.alpha(), max_iter: m.max_iter(), tol: m.tol(), fitted: Some(m) } ;
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
    |m: ElasticNet<f64>| PyElasticNet {
        alpha: m.alpha(), l1_ratio: m.l1_ratio(), max_iter: m.max_iter(), tol: m.tol(), fitted: Some(m),
    } ;
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
#[pyclass(name = "LogisticRegression", module = "tpt_ml_proven", dict)]
pub struct PyLogisticRegression {
    c: f64,
    max_iter: usize,
    tol: f64,
    fitted: Option<LogitState>,
}

impl PyLogisticRegression {
    fn encode(&self) -> PyResult<Vec<u8>> {
        match self.fitted.as_ref().ok_or_else(not_fitted)? {
            LogitState::Binary { model, classes } => {
                labelled_binary_to_bytes(model, *classes).map_err(to_py_err)
            }
            LogitState::Multi(model) => model.to_bytes().map_err(to_py_err),
        }
    }

    fn decode(bytes: &[u8]) -> PyResult<Self> {
        let tag = bytes
            .get(12..16)
            .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
            .ok_or_else(|| PyValueError::new_err("not a tpt-ml-proven model file"))?;
        if tag == kind::LOGISTIC_LABELLED {
            let (model, classes) = labelled_binary_from_bytes(bytes).map_err(to_py_err)?;
            Ok(Self {
                c: model.c(),
                max_iter: model.max_iter(),
                tol: model.tol(),
                fitted: Some(LogitState::Binary { model, classes }),
            })
        } else if tag == kind::MULTICLASS_LOGISTIC {
            let model = MulticlassLogistic::<f64>::from_bytes(bytes).map_err(to_py_err)?;
            Ok(Self {
                c: model.c(),
                max_iter: model.max_iter(),
                tol: model.tol(),
                fitted: Some(LogitState::Multi(model)),
            })
        } else {
            Err(PyValueError::new_err(
                "the file does not hold a LogisticRegression",
            ))
        }
    }
}

#[pymethods]
impl PyLogisticRegression {
    /// Serialises the fitted model to bytes, in the tpt-ml-proven model file format.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &self.encode()?))
    }

    /// Loads a model from bytes written by `to_bytes`. The hyper-parameters come from the file.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        Self::decode(data)
    }

    /// Writes the fitted model to a file.
    fn save(&self, path: &str) -> PyResult<()> {
        std::fs::write(path, self.encode()?).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Loads a model from a file written by `save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let bytes = std::fs::read(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Self::decode(&bytes)
    }

    /// Pickle support: unpickling calls `from_bytes` on the saved bytes.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let py = slf.py();
        let bytes = PyBytes::new(py, &slf.borrow().encode()?);
        let loader = slf.get_type().getattr("from_bytes")?;
        Ok((loader, (bytes,)))
    }

    /// Number of features seen in `fit`. Raises AttributeError before fit, as scikit-learn does.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        (match self.fitted.as_ref() {
            Some(LogitState::Binary { model, .. }) => model.coef().map(|c| c.len()),
            Some(LogitState::Multi(m)) => {
                m.coef().zip(m.intercept()).map(|(c, i)| c.len() / i.len())
            }
            None => None,
        })
        .ok_or_else(|| PyAttributeError::new_err("n_features_in_ is not available before fit"))
    }

    /// scikit-learn's fitted check: true once `fit` has succeeded.
    fn __sklearn_is_fitted__(&self) -> bool {
        self.fitted.is_some()
    }

    /// scikit-learn's tags protocol: what kind of estimator this is.
    fn __sklearn_tags__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        sklearn_tags(py, "classifier")
    }

    /// The constructor parameters, by name.
    #[pyo3(signature = (deep=true))]
    #[allow(unused_variables)]
    fn get_params<'py>(&self, py: Python<'py>, deep: bool) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("C", self.c)?;
        d.set_item("max_iter", self.max_iter)?;
        d.set_item("tol", self.tol)?;
        Ok(d)
    }

    /// Sets constructor parameters by name and returns the estimator. A fitted model keeps its
    /// fitted state until it is refitted.
    #[pyo3(signature = (**params))]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        params: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        if let Some(params) = params {
            let mut this = slf.borrow_mut();
            for (key, value) in params.iter() {
                let key: String = key.extract()?;
                match key.as_str() {
                    "C" => this.c = value.extract()?,
                    "max_iter" => this.max_iter = value.extract()?,
                    "tol" => this.tol = value.extract()?,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter {other:?} for estimator LogisticRegression"
                        )))
                    }
                }
            }
        }
        Ok(slf)
    }

    /// scikit-learn's clone protocol: a new, unfitted estimator with the same parameters.
    fn __sklearn_clone__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let params = slf.borrow().get_params(slf.py(), false)?;
        slf.get_type().call((), Some(&params))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        repr_of("LogisticRegression", &self.get_params(py, false)?)
    }

    /// Fraction of rows of `x` whose predicted label equals the label in `y`.
    fn score(&self, py: Python<'_>, x: &Bound<'_, PyAny>, y: Vec<f64>) -> PyResult<f64> {
        let pred = self.predict(py, x)?;
        tpt_ml_proven_core::metrics::accuracy(&y, &pred).map_err(to_py_err)
    }

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
#[pyclass(name = "StumpForest", module = "tpt_ml_proven", dict)]
pub struct PyStumpForest {
    n_estimators: usize,
    seed: u64,
    fitted: Option<StumpForest<f64>>,
}

impl PyStumpForest {
    fn encode(&self) -> PyResult<Vec<u8>> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .to_bytes()
            .map_err(to_py_err)
    }

    fn decode(bytes: &[u8]) -> PyResult<Self> {
        let m = StumpForest::<f64>::from_bytes(bytes).map_err(to_py_err)?;
        Ok(Self {
            n_estimators: m.n_estimators_setting(),
            seed: m.seed(),
            fitted: Some(m),
        })
    }
}

#[pymethods]
impl PyStumpForest {
    /// Number of features seen in `fit`. Raises AttributeError before fit, as scikit-learn does.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        (self.fitted.as_ref().and_then(|m| m.n_features()))
            .ok_or_else(|| PyAttributeError::new_err("n_features_in_ is not available before fit"))
    }

    /// scikit-learn's fitted check: true once `fit` has succeeded.
    fn __sklearn_is_fitted__(&self) -> bool {
        self.fitted.is_some()
    }

    /// scikit-learn's tags protocol: what kind of estimator this is.
    fn __sklearn_tags__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        sklearn_tags(py, "classifier")
    }

    /// The constructor parameters, by name.
    #[pyo3(signature = (deep=true))]
    #[allow(unused_variables)]
    fn get_params<'py>(&self, py: Python<'py>, deep: bool) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("n_estimators", self.n_estimators)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Sets constructor parameters by name and returns the estimator. A fitted model keeps its
    /// fitted state until it is refitted.
    #[pyo3(signature = (**params))]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        params: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        if let Some(params) = params {
            let mut this = slf.borrow_mut();
            for (key, value) in params.iter() {
                let key: String = key.extract()?;
                match key.as_str() {
                    "n_estimators" => this.n_estimators = value.extract()?,
                    "seed" => this.seed = value.extract()?,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter {other:?} for estimator StumpForest"
                        )))
                    }
                }
            }
        }
        Ok(slf)
    }

    /// scikit-learn's clone protocol: a new, unfitted estimator with the same parameters.
    fn __sklearn_clone__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let params = slf.borrow().get_params(slf.py(), false)?;
        slf.get_type().call((), Some(&params))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        repr_of("StumpForest", &self.get_params(py, false)?)
    }

    /// Fraction of rows of `x` whose predicted label equals the label in `y`.
    fn score(&self, py: Python<'_>, x: &Bound<'_, PyAny>, y: Vec<f64>) -> PyResult<f64> {
        let pred = self.predict(py, x)?;
        tpt_ml_proven_core::metrics::accuracy(&y, &pred).map_err(to_py_err)
    }

    /// Serialises the fitted model to bytes, in the tpt-ml-proven model file format.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &self.encode()?))
    }

    /// Loads a model from bytes written by `to_bytes`. The hyper-parameters come from the file.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        Self::decode(data)
    }

    /// Writes the fitted model to a file.
    fn save(&self, path: &str) -> PyResult<()> {
        std::fs::write(path, self.encode()?).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Loads a model from a file written by `save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let bytes = std::fs::read(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Self::decode(&bytes)
    }

    /// Pickle support: unpickling calls `from_bytes` on the saved bytes.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let py = slf.py();
        let bytes = PyBytes::new(py, &slf.borrow().encode()?);
        let loader = slf.get_type().getattr("from_bytes")?;
        Ok((loader, (bytes,)))
    }

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
