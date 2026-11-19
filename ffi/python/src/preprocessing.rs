// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

use pyo3::exceptions::PyAttributeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};
use tpt_ml_proven_core::{DenseView, Transformer, UnsupervisedEstimator};
use tpt_ml_proven_preprocessing::{
    ImputeStrategy, MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, SimpleImputer,
    StandardScaler,
};

use crate::{matrix_from, not_fitted, repr_of, rows_of, sklearn_tags, to_py_err, to_vec};

/// Strategy names, as scikit-learn spells them.
fn parse_strategy(text: &str) -> PyResult<ImputeStrategy> {
    match text {
        "mean" => Ok(ImputeStrategy::Mean),
        "median" => Ok(ImputeStrategy::Median),
        "most_frequent" => Ok(ImputeStrategy::MostFrequent),
        "constant" => Ok(ImputeStrategy::Constant(0.0)),
        other => Err(PyValueError::new_err(format!(
            "strategy must be 'mean', 'median', 'most_frequent' or 'constant', not {other:?}"
        ))),
    }
}

/// The strategy, with the constant fill value for `constant`.
fn parse_strategy_or(name: &str, fill_value: f64) -> PyResult<ImputeStrategy> {
    match parse_strategy(name)? {
        ImputeStrategy::Constant(_) => Ok(ImputeStrategy::Constant(fill_value)),
        other => Ok(other),
    }
}

fn strategy_name(strategy: ImputeStrategy) -> PyResult<&'static str> {
    Ok(match strategy {
        ImputeStrategy::Mean => "mean",
        ImputeStrategy::Median => "median",
        ImputeStrategy::MostFrequent => "most_frequent",
        ImputeStrategy::Constant(_) => "constant",
    })
}
#[pyclass(name = "StandardScaler", module = "tpt_ml_proven", dict)]
pub struct PyStandardScaler {
    fitted: Option<StandardScaler<f64>>,
}

impl PyStandardScaler {
    fn build(&self) -> StandardScaler<f64> {
        StandardScaler::<f64>::new()
    }

    fn encode(&self) -> PyResult<Vec<u8>> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .to_bytes()
            .map_err(to_py_err)
    }

    fn decode(bytes: &[u8]) -> PyResult<Self> {
        let m = StandardScaler::<f64>::from_bytes(bytes).map_err(to_py_err)?;
        Ok(Self { fitted: Some(m) })
    }
}

#[pymethods]
impl PyStandardScaler {
    /// Number of features seen in `fit`. Raises AttributeError before fit, as scikit-learn does.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        (self.fitted.as_ref().and_then(|m| m.n_features()))
            .ok_or_else(|| PyAttributeError::new_err("n_features_in_ is not available before fit"))
    }

    #[new]
    fn new() -> Self {
        Self { fitted: None }
    }

    /// The constructor parameters, by name. This estimator has none.
    #[pyo3(signature = (deep=true))]
    #[allow(unused_variables)]
    fn get_params<'py>(&self, py: Python<'py>, deep: bool) -> PyResult<Bound<'py, PyDict>> {
        Ok(PyDict::new(py))
    }

    /// Sets constructor parameters by name. This estimator has none, so only an empty call is valid.
    #[pyo3(signature = (**params))]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        params: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        if params.is_some_and(|p| !p.is_empty()) {
            return Err(PyValueError::new_err(format!(
                "invalid parameter for estimator StandardScaler"
            )));
        }
        Ok(slf)
    }

    /// Fits on a 2-D array. The target, if given, is ignored.
    #[pyo3(signature = (x, y=None))]
    #[allow(unused_variables)]
    fn fit<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, Self>> {
        let m = matrix_from(py, x)?;
        let (r, c) = m.shape();
        let data = to_vec::<f64>(&m);
        let mut model = slf.borrow().build();
        UnsupervisedEstimator::fit(
            &mut model,
            DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?,
        )
        .map_err(to_py_err)?;
        slf.borrow_mut().fitted = Some(model);
        Ok(slf)
    }

    /// Fits, then returns the transformed rows.
    #[pyo3(signature = (x, y=None))]
    #[allow(unused_variables)]
    fn fit_transform(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: Option<Vec<f64>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let m = matrix_from(py, x)?;
        let (r, c) = m.shape();
        let data = to_vec::<f64>(&m);
        let mut model = self.build();
        let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
        let out = Transformer::fit_transform(&mut model, view).map_err(to_py_err)?;
        self.fitted = Some(model);
        Ok(rows_of(&out))
    }

    /// The transformed rows of `x`.
    fn transform(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let mm = matrix_from(py, x)?;
        let (r, c) = mm.shape();
        let data = to_vec::<f64>(&mm);
        let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
        let out = Transformer::transform(m, view).map_err(to_py_err)?;
        Ok(rows_of(&out))
    }

    /// scikit-learn's clone protocol: a new, unfitted estimator with the same parameters.
    fn __sklearn_clone__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let params = slf.borrow().get_params(slf.py(), false)?;
        slf.get_type().call((), Some(&params))
    }

    /// scikit-learn's tags protocol: what kind of estimator this is.
    fn __sklearn_tags__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        sklearn_tags(py, "transformer")
    }

    /// scikit-learn's fitted check: true once `fit` has succeeded.
    fn __sklearn_is_fitted__(&self) -> bool {
        self.fitted.is_some()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        repr_of("StandardScaler", &self.get_params(py, false)?)
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
}

#[pyclass(name = "MinMaxScaler", module = "tpt_ml_proven", dict)]
pub struct PyMinMaxScaler {
    fitted: Option<MinMaxScaler<f64>>,
}

impl PyMinMaxScaler {
    fn build(&self) -> MinMaxScaler<f64> {
        MinMaxScaler::<f64>::new()
    }

    fn encode(&self) -> PyResult<Vec<u8>> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .to_bytes()
            .map_err(to_py_err)
    }

    fn decode(bytes: &[u8]) -> PyResult<Self> {
        let m = MinMaxScaler::<f64>::from_bytes(bytes).map_err(to_py_err)?;
        Ok(Self { fitted: Some(m) })
    }
}

#[pymethods]
impl PyMinMaxScaler {
    /// Number of features seen in `fit`. Raises AttributeError before fit, as scikit-learn does.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        (self.fitted.as_ref().and_then(|m| m.n_features()))
            .ok_or_else(|| PyAttributeError::new_err("n_features_in_ is not available before fit"))
    }

    #[new]
    fn new() -> Self {
        Self { fitted: None }
    }

    /// The constructor parameters, by name. This estimator has none.
    #[pyo3(signature = (deep=true))]
    #[allow(unused_variables)]
    fn get_params<'py>(&self, py: Python<'py>, deep: bool) -> PyResult<Bound<'py, PyDict>> {
        Ok(PyDict::new(py))
    }

    /// Sets constructor parameters by name. This estimator has none, so only an empty call is valid.
    #[pyo3(signature = (**params))]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        params: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        if params.is_some_and(|p| !p.is_empty()) {
            return Err(PyValueError::new_err(format!(
                "invalid parameter for estimator MinMaxScaler"
            )));
        }
        Ok(slf)
    }

    /// Fits on a 2-D array. The target, if given, is ignored.
    #[pyo3(signature = (x, y=None))]
    #[allow(unused_variables)]
    fn fit<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, Self>> {
        let m = matrix_from(py, x)?;
        let (r, c) = m.shape();
        let data = to_vec::<f64>(&m);
        let mut model = slf.borrow().build();
        UnsupervisedEstimator::fit(
            &mut model,
            DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?,
        )
        .map_err(to_py_err)?;
        slf.borrow_mut().fitted = Some(model);
        Ok(slf)
    }

    /// Fits, then returns the transformed rows.
    #[pyo3(signature = (x, y=None))]
    #[allow(unused_variables)]
    fn fit_transform(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: Option<Vec<f64>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let m = matrix_from(py, x)?;
        let (r, c) = m.shape();
        let data = to_vec::<f64>(&m);
        let mut model = self.build();
        let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
        let out = Transformer::fit_transform(&mut model, view).map_err(to_py_err)?;
        self.fitted = Some(model);
        Ok(rows_of(&out))
    }

    /// The transformed rows of `x`.
    fn transform(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let mm = matrix_from(py, x)?;
        let (r, c) = mm.shape();
        let data = to_vec::<f64>(&mm);
        let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
        let out = Transformer::transform(m, view).map_err(to_py_err)?;
        Ok(rows_of(&out))
    }

    /// scikit-learn's clone protocol: a new, unfitted estimator with the same parameters.
    fn __sklearn_clone__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let params = slf.borrow().get_params(slf.py(), false)?;
        slf.get_type().call((), Some(&params))
    }

    /// scikit-learn's tags protocol: what kind of estimator this is.
    fn __sklearn_tags__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        sklearn_tags(py, "transformer")
    }

    /// scikit-learn's fitted check: true once `fit` has succeeded.
    fn __sklearn_is_fitted__(&self) -> bool {
        self.fitted.is_some()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        repr_of("MinMaxScaler", &self.get_params(py, false)?)
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
}

#[pyclass(name = "RobustScaler", module = "tpt_ml_proven", dict)]
pub struct PyRobustScaler {
    fitted: Option<RobustScaler<f64>>,
}

impl PyRobustScaler {
    fn build(&self) -> RobustScaler<f64> {
        RobustScaler::<f64>::new()
    }

    fn encode(&self) -> PyResult<Vec<u8>> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .to_bytes()
            .map_err(to_py_err)
    }

    fn decode(bytes: &[u8]) -> PyResult<Self> {
        let m = RobustScaler::<f64>::from_bytes(bytes).map_err(to_py_err)?;
        Ok(Self { fitted: Some(m) })
    }
}

#[pymethods]
impl PyRobustScaler {
    /// Number of features seen in `fit`. Raises AttributeError before fit, as scikit-learn does.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        (self.fitted.as_ref().and_then(|m| m.n_features()))
            .ok_or_else(|| PyAttributeError::new_err("n_features_in_ is not available before fit"))
    }

    #[new]
    fn new() -> Self {
        Self { fitted: None }
    }

    /// The constructor parameters, by name. This estimator has none.
    #[pyo3(signature = (deep=true))]
    #[allow(unused_variables)]
    fn get_params<'py>(&self, py: Python<'py>, deep: bool) -> PyResult<Bound<'py, PyDict>> {
        Ok(PyDict::new(py))
    }

    /// Sets constructor parameters by name. This estimator has none, so only an empty call is valid.
    #[pyo3(signature = (**params))]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        params: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        if params.is_some_and(|p| !p.is_empty()) {
            return Err(PyValueError::new_err(format!(
                "invalid parameter for estimator RobustScaler"
            )));
        }
        Ok(slf)
    }

    /// Fits on a 2-D array. The target, if given, is ignored.
    #[pyo3(signature = (x, y=None))]
    #[allow(unused_variables)]
    fn fit<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, Self>> {
        let m = matrix_from(py, x)?;
        let (r, c) = m.shape();
        let data = to_vec::<f64>(&m);
        let mut model = slf.borrow().build();
        UnsupervisedEstimator::fit(
            &mut model,
            DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?,
        )
        .map_err(to_py_err)?;
        slf.borrow_mut().fitted = Some(model);
        Ok(slf)
    }

    /// Fits, then returns the transformed rows.
    #[pyo3(signature = (x, y=None))]
    #[allow(unused_variables)]
    fn fit_transform(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: Option<Vec<f64>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let m = matrix_from(py, x)?;
        let (r, c) = m.shape();
        let data = to_vec::<f64>(&m);
        let mut model = self.build();
        let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
        let out = Transformer::fit_transform(&mut model, view).map_err(to_py_err)?;
        self.fitted = Some(model);
        Ok(rows_of(&out))
    }

    /// The transformed rows of `x`.
    fn transform(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let mm = matrix_from(py, x)?;
        let (r, c) = mm.shape();
        let data = to_vec::<f64>(&mm);
        let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
        let out = Transformer::transform(m, view).map_err(to_py_err)?;
        Ok(rows_of(&out))
    }

    /// scikit-learn's clone protocol: a new, unfitted estimator with the same parameters.
    fn __sklearn_clone__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let params = slf.borrow().get_params(slf.py(), false)?;
        slf.get_type().call((), Some(&params))
    }

    /// scikit-learn's tags protocol: what kind of estimator this is.
    fn __sklearn_tags__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        sklearn_tags(py, "transformer")
    }

    /// scikit-learn's fitted check: true once `fit` has succeeded.
    fn __sklearn_is_fitted__(&self) -> bool {
        self.fitted.is_some()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        repr_of("RobustScaler", &self.get_params(py, false)?)
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
}

#[pyclass(name = "OneHotEncoder", module = "tpt_ml_proven", dict)]
pub struct PyOneHotEncoder {
    fitted: Option<OneHotEncoder<f64>>,
}

impl PyOneHotEncoder {
    fn build(&self) -> OneHotEncoder<f64> {
        OneHotEncoder::<f64>::new()
    }

    fn encode(&self) -> PyResult<Vec<u8>> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .to_bytes()
            .map_err(to_py_err)
    }

    fn decode(bytes: &[u8]) -> PyResult<Self> {
        let m = OneHotEncoder::<f64>::from_bytes(bytes).map_err(to_py_err)?;
        Ok(Self { fitted: Some(m) })
    }
}

#[pymethods]
impl PyOneHotEncoder {
    /// Number of features seen in `fit`. Raises AttributeError before fit, as scikit-learn does.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        (self
            .fitted
            .as_ref()
            .and_then(|m| m.categories().map(|c| c.len())))
        .ok_or_else(|| PyAttributeError::new_err("n_features_in_ is not available before fit"))
    }

    #[new]
    fn new() -> Self {
        Self { fitted: None }
    }

    /// The constructor parameters, by name. This estimator has none.
    #[pyo3(signature = (deep=true))]
    #[allow(unused_variables)]
    fn get_params<'py>(&self, py: Python<'py>, deep: bool) -> PyResult<Bound<'py, PyDict>> {
        Ok(PyDict::new(py))
    }

    /// Sets constructor parameters by name. This estimator has none, so only an empty call is valid.
    #[pyo3(signature = (**params))]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        params: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        if params.is_some_and(|p| !p.is_empty()) {
            return Err(PyValueError::new_err(format!(
                "invalid parameter for estimator OneHotEncoder"
            )));
        }
        Ok(slf)
    }

    /// Fits on a 2-D array. The target, if given, is ignored.
    #[pyo3(signature = (x, y=None))]
    #[allow(unused_variables)]
    fn fit<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, Self>> {
        let m = matrix_from(py, x)?;
        let (r, c) = m.shape();
        let data = to_vec::<f64>(&m);
        let mut model = slf.borrow().build();
        UnsupervisedEstimator::fit(
            &mut model,
            DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?,
        )
        .map_err(to_py_err)?;
        slf.borrow_mut().fitted = Some(model);
        Ok(slf)
    }

    /// Fits, then returns the transformed rows.
    #[pyo3(signature = (x, y=None))]
    #[allow(unused_variables)]
    fn fit_transform(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: Option<Vec<f64>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let m = matrix_from(py, x)?;
        let (r, c) = m.shape();
        let data = to_vec::<f64>(&m);
        let mut model = self.build();
        let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
        let out = Transformer::fit_transform(&mut model, view).map_err(to_py_err)?;
        self.fitted = Some(model);
        Ok(rows_of(&out))
    }

    /// The transformed rows of `x`.
    fn transform(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let mm = matrix_from(py, x)?;
        let (r, c) = mm.shape();
        let data = to_vec::<f64>(&mm);
        let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
        let out = Transformer::transform(m, view).map_err(to_py_err)?;
        Ok(rows_of(&out))
    }

    /// scikit-learn's clone protocol: a new, unfitted estimator with the same parameters.
    fn __sklearn_clone__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let params = slf.borrow().get_params(slf.py(), false)?;
        slf.get_type().call((), Some(&params))
    }

    /// scikit-learn's tags protocol: what kind of estimator this is.
    fn __sklearn_tags__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        sklearn_tags(py, "transformer")
    }

    /// scikit-learn's fitted check: true once `fit` has succeeded.
    fn __sklearn_is_fitted__(&self) -> bool {
        self.fitted.is_some()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        repr_of("OneHotEncoder", &self.get_params(py, false)?)
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
}

#[pyclass(name = "OrdinalEncoder", module = "tpt_ml_proven", dict)]
pub struct PyOrdinalEncoder {
    fitted: Option<OrdinalEncoder<f64>>,
}

impl PyOrdinalEncoder {
    fn build(&self) -> OrdinalEncoder<f64> {
        OrdinalEncoder::<f64>::new()
    }

    fn encode(&self) -> PyResult<Vec<u8>> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .to_bytes()
            .map_err(to_py_err)
    }

    fn decode(bytes: &[u8]) -> PyResult<Self> {
        let m = OrdinalEncoder::<f64>::from_bytes(bytes).map_err(to_py_err)?;
        Ok(Self { fitted: Some(m) })
    }
}

#[pymethods]
impl PyOrdinalEncoder {
    /// Number of features seen in `fit`. Raises AttributeError before fit, as scikit-learn does.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        (self
            .fitted
            .as_ref()
            .and_then(|m| m.categories().map(|c| c.len())))
        .ok_or_else(|| PyAttributeError::new_err("n_features_in_ is not available before fit"))
    }

    #[new]
    fn new() -> Self {
        Self { fitted: None }
    }

    /// The constructor parameters, by name. This estimator has none.
    #[pyo3(signature = (deep=true))]
    #[allow(unused_variables)]
    fn get_params<'py>(&self, py: Python<'py>, deep: bool) -> PyResult<Bound<'py, PyDict>> {
        Ok(PyDict::new(py))
    }

    /// Sets constructor parameters by name. This estimator has none, so only an empty call is valid.
    #[pyo3(signature = (**params))]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        params: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        if params.is_some_and(|p| !p.is_empty()) {
            return Err(PyValueError::new_err(format!(
                "invalid parameter for estimator OrdinalEncoder"
            )));
        }
        Ok(slf)
    }

    /// Fits on a 2-D array. The target, if given, is ignored.
    #[pyo3(signature = (x, y=None))]
    #[allow(unused_variables)]
    fn fit<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, Self>> {
        let m = matrix_from(py, x)?;
        let (r, c) = m.shape();
        let data = to_vec::<f64>(&m);
        let mut model = slf.borrow().build();
        UnsupervisedEstimator::fit(
            &mut model,
            DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?,
        )
        .map_err(to_py_err)?;
        slf.borrow_mut().fitted = Some(model);
        Ok(slf)
    }

    /// Fits, then returns the transformed rows.
    #[pyo3(signature = (x, y=None))]
    #[allow(unused_variables)]
    fn fit_transform(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: Option<Vec<f64>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let m = matrix_from(py, x)?;
        let (r, c) = m.shape();
        let data = to_vec::<f64>(&m);
        let mut model = self.build();
        let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
        let out = Transformer::fit_transform(&mut model, view).map_err(to_py_err)?;
        self.fitted = Some(model);
        Ok(rows_of(&out))
    }

    /// The transformed rows of `x`.
    fn transform(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let mm = matrix_from(py, x)?;
        let (r, c) = mm.shape();
        let data = to_vec::<f64>(&mm);
        let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
        let out = Transformer::transform(m, view).map_err(to_py_err)?;
        Ok(rows_of(&out))
    }

    /// scikit-learn's clone protocol: a new, unfitted estimator with the same parameters.
    fn __sklearn_clone__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let params = slf.borrow().get_params(slf.py(), false)?;
        slf.get_type().call((), Some(&params))
    }

    /// scikit-learn's tags protocol: what kind of estimator this is.
    fn __sklearn_tags__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        sklearn_tags(py, "transformer")
    }

    /// scikit-learn's fitted check: true once `fit` has succeeded.
    fn __sklearn_is_fitted__(&self) -> bool {
        self.fitted.is_some()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        repr_of("OrdinalEncoder", &self.get_params(py, false)?)
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
}

#[pyclass(name = "SimpleImputer", module = "tpt_ml_proven", dict)]
pub struct PySimpleImputer {
    strategy: String,
    fill_value: f64,
    fitted: Option<SimpleImputer<f64>>,
}

impl PySimpleImputer {
    fn build(&self) -> SimpleImputer<f64> {
        SimpleImputer::<f64>::new(
            parse_strategy_or(&self.strategy, self.fill_value).expect("validated"),
        )
    }

    fn encode(&self) -> PyResult<Vec<u8>> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .to_bytes()
            .map_err(to_py_err)
    }

    fn decode(bytes: &[u8]) -> PyResult<Self> {
        let m = SimpleImputer::<f64>::from_bytes(bytes).map_err(to_py_err)?;
        Ok(Self {
            strategy: strategy_name(m.strategy())?.to_string(),
            fill_value: match m.strategy() {
                ImputeStrategy::Constant(v) => v,
                _ => 0.0,
            },
            fitted: Some(m),
        })
    }
}

#[pymethods]
impl PySimpleImputer {
    /// Number of features seen in `fit`. Raises AttributeError before fit, as scikit-learn does.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        (self
            .fitted
            .as_ref()
            .and_then(|m| m.statistics().map(|s| s.len())))
        .ok_or_else(|| PyAttributeError::new_err("n_features_in_ is not available before fit"))
    }

    /// Constructor parameters, as scikit-learn names them.
    #[new]
    #[pyo3(signature = (strategy="mean", fill_value=0.0))]
    fn new(strategy: &str, fill_value: f64) -> PyResult<Self> {
        parse_strategy(strategy)?;
        Ok(Self {
            strategy: strategy.to_string(),
            fill_value,
            fitted: None,
        })
    }

    /// The constructor parameters, by name.
    #[pyo3(signature = (deep=true))]
    #[allow(unused_variables)]
    fn get_params<'py>(&self, py: Python<'py>, deep: bool) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("strategy", self.strategy.as_str())?;
        d.set_item("fill_value", self.fill_value)?;
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
                    "strategy" => this.strategy = value.extract()?,
                    "fill_value" => this.fill_value = value.extract()?,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter {other:?} for estimator SimpleImputer"
                        )))
                    }
                }
            }
        }
        Ok(slf)
    }

    /// Fits on a 2-D array. The target, if given, is ignored.
    #[pyo3(signature = (x, y=None))]
    #[allow(unused_variables)]
    fn fit<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, Self>> {
        let m = matrix_from(py, x)?;
        let (r, c) = m.shape();
        let data = to_vec::<f64>(&m);
        let mut model = slf.borrow().build();
        UnsupervisedEstimator::fit(
            &mut model,
            DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?,
        )
        .map_err(to_py_err)?;
        slf.borrow_mut().fitted = Some(model);
        Ok(slf)
    }

    /// Fits, then returns the transformed rows.
    #[pyo3(signature = (x, y=None))]
    #[allow(unused_variables)]
    fn fit_transform(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: Option<Vec<f64>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let m = matrix_from(py, x)?;
        let (r, c) = m.shape();
        let data = to_vec::<f64>(&m);
        let mut model = self.build();
        let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
        let out = Transformer::fit_transform(&mut model, view).map_err(to_py_err)?;
        self.fitted = Some(model);
        Ok(rows_of(&out))
    }

    /// The transformed rows of `x`.
    fn transform(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let mm = matrix_from(py, x)?;
        let (r, c) = mm.shape();
        let data = to_vec::<f64>(&mm);
        let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
        let out = Transformer::transform(m, view).map_err(to_py_err)?;
        Ok(rows_of(&out))
    }

    /// scikit-learn's clone protocol: a new, unfitted estimator with the same parameters.
    fn __sklearn_clone__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let params = slf.borrow().get_params(slf.py(), false)?;
        slf.get_type().call((), Some(&params))
    }

    /// scikit-learn's tags protocol: what kind of estimator this is.
    fn __sklearn_tags__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        sklearn_tags(py, "transformer")
    }

    /// scikit-learn's fitted check: true once `fit` has succeeded.
    fn __sklearn_is_fitted__(&self) -> bool {
        self.fitted.is_some()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        repr_of("SimpleImputer", &self.get_params(py, false)?)
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
}
