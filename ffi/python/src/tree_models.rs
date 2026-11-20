// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

use pyo3::exceptions::PyAttributeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};
use tpt_ml_proven_core::metrics::{accuracy, r2};
use tpt_ml_proven_core::{DenseView, Predictor, SupervisedEstimator};
use tpt_ml_proven_trees::{
    Criterion, DecisionTreeClassifier, DecisionTreeRegressor, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
};

use crate::{matrix_from, not_fitted, repr_of, sklearn_tags, to_py_err, to_vec};

/// `"gini"` or `"entropy"`.
fn parse_criterion(text: &str) -> PyResult<Criterion> {
    match text {
        "gini" => Ok(Criterion::Gini),
        "entropy" => Ok(Criterion::Entropy),
        other => Err(PyValueError::new_err(format!(
            "criterion must be 'gini' or 'entropy', not {other:?}"
        ))),
    }
}

fn criterion_name(c: Criterion) -> &'static str {
    match c {
        Criterion::Gini => "gini",
        Criterion::Entropy => "entropy",
    }
}
#[pyclass(name = "DecisionTreeRegressor", module = "tpt_ml_proven", dict)]
pub struct PyDecisionTreeRegressor {
    max_depth: usize,
    min_samples_leaf: usize,
    min_samples_split: usize,
    max_leaf_nodes: Option<usize>,
    fitted: Option<DecisionTreeRegressor>,
}

impl PyDecisionTreeRegressor {
    fn build(&self) -> DecisionTreeRegressor {
        DecisionTreeRegressor::new(self.max_depth)
            .with_min_samples_leaf(self.min_samples_leaf)
            .with_min_samples_split(self.min_samples_split)
            .with_max_leaf_nodes(self.max_leaf_nodes)
    }

    fn encode(&self) -> PyResult<Vec<u8>> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .to_bytes()
            .map_err(to_py_err)
    }

    fn decode(bytes: &[u8]) -> PyResult<Self> {
        let m = DecisionTreeRegressor::from_bytes(bytes).map_err(to_py_err)?;
        Ok(Self {
            max_depth: m.max_depth(),
            min_samples_leaf: m.min_samples_leaf(),
            min_samples_split: m.min_samples_split(),
            max_leaf_nodes: m.max_leaf_nodes(),
            fitted: Some(m),
        })
    }
}

#[pymethods]
impl PyDecisionTreeRegressor {
    /// Number of features seen in `fit`. Raises AttributeError before fit, as scikit-learn does.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        (self
            .fitted
            .as_ref()
            .and_then(|m| m.tree().map(|t| t.n_features())))
        .ok_or_else(|| PyAttributeError::new_err("n_features_in_ is not available before fit"))
    }

    /// Each feature's share of the total squared-error reduction. Raises AttributeError before
    /// fit, and for a model loaded from a file written before importances were stored.
    #[getter]
    fn feature_importances_(&self) -> PyResult<Vec<f64>> {
        self.fitted
            .as_ref()
            .and_then(|m| m.feature_importances())
            .ok_or_else(|| {
                PyAttributeError::new_err("feature_importances_ is not available for this model")
            })
    }

    /// Constructor parameters, as scikit-learn names them.
    #[new]
    #[pyo3(signature = (max_depth=3, *, min_samples_leaf=1, min_samples_split=2, max_leaf_nodes=None))]
    fn new(
        max_depth: usize,
        min_samples_leaf: usize,
        min_samples_split: usize,
        max_leaf_nodes: Option<usize>,
    ) -> PyResult<Self> {
        Ok(Self {
            max_depth,
            min_samples_leaf,
            min_samples_split,
            max_leaf_nodes,
            fitted: None,
        })
    }

    /// The constructor parameters, by name.
    #[pyo3(signature = (deep=true))]
    #[allow(unused_variables)]
    fn get_params<'py>(&self, py: Python<'py>, deep: bool) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("max_depth", self.max_depth)?;
        d.set_item("min_samples_leaf", self.min_samples_leaf)?;
        d.set_item("min_samples_split", self.min_samples_split)?;
        d.set_item("max_leaf_nodes", self.max_leaf_nodes)?;
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
                    "max_depth" => this.max_depth = value.extract()?,
                    "min_samples_leaf" => this.min_samples_leaf = value.extract()?,
                    "min_samples_split" => this.min_samples_split = value.extract()?,
                    "max_leaf_nodes" => this.max_leaf_nodes = value.extract()?,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter {other:?} for estimator DecisionTreeRegressor"
                        )))
                    }
                }
            }
        }
        Ok(slf)
    }

    /// Fits to a 2-D array and a target sequence, and returns the estimator.
    fn fit<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: Vec<f64>,
    ) -> PyResult<Bound<'py, Self>> {
        let m = matrix_from(py, x)?;
        let (r, c) = m.shape();
        let data = to_vec::<f64>(&m);
        if y.len() != r {
            return Err(PyValueError::new_err(
                "the target has a different length from the number of rows of X",
            ));
        }
        let mut model = slf.borrow().build();
        SupervisedEstimator::fit(
            &mut model,
            DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?,
            &y[..],
        )
        .map_err(to_py_err)?;
        slf.borrow_mut().fitted = Some(model);
        Ok(slf)
    }

    /// The predicted value (or label) for each row.
    fn predict(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let mm = matrix_from(py, x)?;
        let (r, c) = mm.shape();
        let data = to_vec::<f64>(&mm);
        let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
        Predictor::predict(m, view).map_err(to_py_err)
    }

    /// Fraction of rows whose predicted label equals the label in `y` (classifiers) or the R²
    /// of the predictions (regressors).
    fn score(&self, py: Python<'_>, x: &Bound<'_, PyAny>, y: Vec<f64>) -> PyResult<f64> {
        let pred = self.predict(py, x)?;
        r2(&y, &pred).map_err(to_py_err)
    }

    /// scikit-learn's clone protocol: a new, unfitted estimator with the same parameters.
    fn __sklearn_clone__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let params = slf.borrow().get_params(slf.py(), false)?;
        slf.get_type().call((), Some(&params))
    }

    /// scikit-learn's tags protocol: what kind of estimator this is.
    fn __sklearn_tags__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        sklearn_tags(py, "regressor")
    }

    /// scikit-learn's fitted check: true once `fit` has succeeded.
    fn __sklearn_is_fitted__(&self) -> bool {
        self.fitted.is_some()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        repr_of("DecisionTreeRegressor", &self.get_params(py, false)?)
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

#[pyclass(name = "DecisionTreeClassifier", module = "tpt_ml_proven", dict)]
pub struct PyDecisionTreeClassifier {
    max_depth: usize,
    min_samples_leaf: usize,
    min_samples_split: usize,
    max_leaf_nodes: Option<usize>,
    criterion: String,
    fitted: Option<DecisionTreeClassifier>,
}

impl PyDecisionTreeClassifier {
    fn build(&self) -> DecisionTreeClassifier {
        DecisionTreeClassifier::new(self.max_depth)
            .with_min_samples_leaf(self.min_samples_leaf)
            .with_min_samples_split(self.min_samples_split)
            .with_max_leaf_nodes(self.max_leaf_nodes)
            .with_criterion(parse_criterion(&self.criterion).expect("validated"))
    }

    fn encode(&self) -> PyResult<Vec<u8>> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .to_bytes()
            .map_err(to_py_err)
    }

    fn decode(bytes: &[u8]) -> PyResult<Self> {
        let m = DecisionTreeClassifier::from_bytes(bytes).map_err(to_py_err)?;
        Ok(Self {
            max_depth: m.max_depth(),
            min_samples_leaf: m.min_samples_leaf(),
            min_samples_split: m.min_samples_split(),
            max_leaf_nodes: m.max_leaf_nodes(),
            criterion: criterion_name(m.criterion()).to_string(),
            fitted: Some(m),
        })
    }
}

#[pymethods]
impl PyDecisionTreeClassifier {
    /// Number of features seen in `fit`. Raises AttributeError before fit, as scikit-learn does.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        (self.fitted.as_ref().and_then(|m| m.n_features()))
            .ok_or_else(|| PyAttributeError::new_err("n_features_in_ is not available before fit"))
    }

    /// Each feature's share of the total impurity reduction. Raises AttributeError before fit,
    /// and for a model loaded from a file written before importances were stored.
    #[getter]
    fn feature_importances_(&self) -> PyResult<Vec<f64>> {
        self.fitted
            .as_ref()
            .and_then(|m| m.feature_importances())
            .ok_or_else(|| {
                PyAttributeError::new_err("feature_importances_ is not available for this model")
            })
    }

    /// Constructor parameters, as scikit-learn names them.
    #[new]
    #[pyo3(signature = (max_depth=3, *, min_samples_leaf=1, min_samples_split=2, max_leaf_nodes=None, criterion="gini"))]
    fn new(
        max_depth: usize,
        min_samples_leaf: usize,
        min_samples_split: usize,
        max_leaf_nodes: Option<usize>,
        criterion: &str,
    ) -> PyResult<Self> {
        parse_criterion(criterion)?;
        Ok(Self {
            max_depth,
            min_samples_leaf,
            min_samples_split,
            max_leaf_nodes,
            criterion: criterion.to_string(),
            fitted: None,
        })
    }

    /// The constructor parameters, by name.
    #[pyo3(signature = (deep=true))]
    #[allow(unused_variables)]
    fn get_params<'py>(&self, py: Python<'py>, deep: bool) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("max_depth", self.max_depth)?;
        d.set_item("min_samples_leaf", self.min_samples_leaf)?;
        d.set_item("min_samples_split", self.min_samples_split)?;
        d.set_item("max_leaf_nodes", self.max_leaf_nodes)?;
        d.set_item("criterion", self.criterion.as_str())?;
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
                    "max_depth" => this.max_depth = value.extract()?,
                    "min_samples_leaf" => this.min_samples_leaf = value.extract()?,
                    "min_samples_split" => this.min_samples_split = value.extract()?,
                    "max_leaf_nodes" => this.max_leaf_nodes = value.extract()?,
                    "criterion" => this.criterion = value.extract()?,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter {other:?} for estimator DecisionTreeClassifier"
                        )))
                    }
                }
            }
        }
        Ok(slf)
    }

    /// Fits to a 2-D array and a target sequence, and returns the estimator.
    fn fit<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: Vec<f64>,
    ) -> PyResult<Bound<'py, Self>> {
        let m = matrix_from(py, x)?;
        let (r, c) = m.shape();
        let data = to_vec::<f64>(&m);
        if y.len() != r {
            return Err(PyValueError::new_err(
                "the target has a different length from the number of rows of X",
            ));
        }
        let mut model = slf.borrow().build();
        SupervisedEstimator::fit(
            &mut model,
            DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?,
            &y[..],
        )
        .map_err(to_py_err)?;
        slf.borrow_mut().fitted = Some(model);
        Ok(slf)
    }

    /// The predicted value (or label) for each row.
    fn predict(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let mm = matrix_from(py, x)?;
        let (r, c) = mm.shape();
        let data = to_vec::<f64>(&mm);
        let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
        Predictor::predict(m, view).map_err(to_py_err)
    }

    /// Fraction of rows whose predicted label equals the label in `y` (classifiers) or the R²
    /// of the predictions (regressors).
    fn score(&self, py: Python<'_>, x: &Bound<'_, PyAny>, y: Vec<f64>) -> PyResult<f64> {
        let pred = self.predict(py, x)?;
        accuracy(&y, &pred).map_err(to_py_err)
    }

    /// scikit-learn's clone protocol: a new, unfitted estimator with the same parameters.
    fn __sklearn_clone__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let params = slf.borrow().get_params(slf.py(), false)?;
        slf.get_type().call((), Some(&params))
    }

    /// scikit-learn's tags protocol: what kind of estimator this is.
    fn __sklearn_tags__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        sklearn_tags(py, "classifier")
    }

    /// scikit-learn's fitted check: true once `fit` has succeeded.
    fn __sklearn_is_fitted__(&self) -> bool {
        self.fitted.is_some()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        repr_of("DecisionTreeClassifier", &self.get_params(py, false)?)
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

    /// The distinct class labels seen in training, ascending.
    #[getter]
    fn classes_(&self) -> PyResult<Vec<f64>> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        m.classes().map(|c| c.to_vec()).ok_or_else(not_fitted)
    }
}

#[pyclass(name = "GradientBoostingRegressor", module = "tpt_ml_proven", dict)]
pub struct PyGradientBoostingRegressor {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_leaf: usize,
    min_samples_split: usize,
    max_leaf_nodes: Option<usize>,
    subsample: f64,
    max_features: Option<usize>,
    seed: u64,
    fitted: Option<GradientBoostingRegressor>,
}

impl PyGradientBoostingRegressor {
    fn build(&self) -> GradientBoostingRegressor {
        {
            let mut m = GradientBoostingRegressor::new(self.n_estimators, self.max_depth)
                .with_learning_rate(self.learning_rate)
                .with_min_samples_leaf(self.min_samples_leaf)
                .with_min_samples_split(self.min_samples_split)
                .with_max_leaf_nodes(self.max_leaf_nodes)
                .with_subsample(self.subsample)
                .with_seed(self.seed);
            if let Some(f) = self.max_features {
                m = m.with_max_features(f);
            }
            m
        }
    }

    fn encode(&self) -> PyResult<Vec<u8>> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .to_bytes()
            .map_err(to_py_err)
    }

    fn decode(bytes: &[u8]) -> PyResult<Self> {
        let m = GradientBoostingRegressor::from_bytes(bytes).map_err(to_py_err)?;
        Ok(Self {
            n_estimators: m.n_estimators(),
            learning_rate: m.learning_rate(),
            max_depth: m.max_depth(),
            min_samples_leaf: m.min_samples_leaf(),
            min_samples_split: m.min_samples_split(),
            max_leaf_nodes: m.max_leaf_nodes(),
            subsample: m.subsample(),
            max_features: m.max_features(),
            seed: m.seed(),
            fitted: Some(m),
        })
    }
}

#[pymethods]
impl PyGradientBoostingRegressor {
    /// Number of features seen in `fit`. Raises AttributeError before fit, as scikit-learn does.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        (self
            .fitted
            .as_ref()
            .and_then(|m| m.trees().and_then(|t| t.first()).map(|t| t.n_features())))
        .ok_or_else(|| PyAttributeError::new_err("n_features_in_ is not available before fit"))
    }

    /// Each feature's share of the total squared-error reduction over all rounds. Raises
    /// AttributeError before fit, and for a model loaded from a file written before importances
    /// were stored.
    #[getter]
    fn feature_importances_(&self) -> PyResult<Vec<f64>> {
        self.fitted
            .as_ref()
            .and_then(|m| m.feature_importances())
            .ok_or_else(|| {
                PyAttributeError::new_err("feature_importances_ is not available for this model")
            })
    }

    /// Constructor parameters, as scikit-learn names them.
    #[new]
    #[pyo3(signature = (n_estimators=100, *, learning_rate=0.1, max_depth=3, min_samples_leaf=1, min_samples_split=2, max_leaf_nodes=None, subsample=1.0, max_features=None, seed=0))]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_samples_split: usize,
        max_leaf_nodes: Option<usize>,
        subsample: f64,
        max_features: Option<usize>,
        seed: u64,
    ) -> PyResult<Self> {
        Ok(Self {
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            max_leaf_nodes,
            subsample,
            max_features,
            seed,
            fitted: None,
        })
    }

    /// The constructor parameters, by name.
    #[pyo3(signature = (deep=true))]
    #[allow(unused_variables)]
    fn get_params<'py>(&self, py: Python<'py>, deep: bool) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("n_estimators", self.n_estimators)?;
        d.set_item("learning_rate", self.learning_rate)?;
        d.set_item("max_depth", self.max_depth)?;
        d.set_item("min_samples_leaf", self.min_samples_leaf)?;
        d.set_item("min_samples_split", self.min_samples_split)?;
        d.set_item("max_leaf_nodes", self.max_leaf_nodes)?;
        d.set_item("subsample", self.subsample)?;
        d.set_item("max_features", self.max_features)?;
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
                    "learning_rate" => this.learning_rate = value.extract()?,
                    "max_depth" => this.max_depth = value.extract()?,
                    "min_samples_leaf" => this.min_samples_leaf = value.extract()?,
                    "min_samples_split" => this.min_samples_split = value.extract()?,
                    "max_leaf_nodes" => this.max_leaf_nodes = value.extract()?,
                    "subsample" => this.subsample = value.extract()?,
                    "max_features" => this.max_features = value.extract()?,
                    "seed" => this.seed = value.extract()?,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter {other:?} for estimator GradientBoostingRegressor"
                        )))
                    }
                }
            }
        }
        Ok(slf)
    }

    /// Fits to a 2-D array and a target sequence, and returns the estimator.
    fn fit<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: Vec<f64>,
    ) -> PyResult<Bound<'py, Self>> {
        let m = matrix_from(py, x)?;
        let (r, c) = m.shape();
        let data = to_vec::<f64>(&m);
        if y.len() != r {
            return Err(PyValueError::new_err(
                "the target has a different length from the number of rows of X",
            ));
        }
        let mut model = slf.borrow().build();
        SupervisedEstimator::fit(
            &mut model,
            DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?,
            &y[..],
        )
        .map_err(to_py_err)?;
        slf.borrow_mut().fitted = Some(model);
        Ok(slf)
    }

    /// The predicted value (or label) for each row.
    fn predict(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let mm = matrix_from(py, x)?;
        let (r, c) = mm.shape();
        let data = to_vec::<f64>(&mm);
        let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
        Predictor::predict(m, view).map_err(to_py_err)
    }

    /// Fraction of rows whose predicted label equals the label in `y` (classifiers) or the R²
    /// of the predictions (regressors).
    fn score(&self, py: Python<'_>, x: &Bound<'_, PyAny>, y: Vec<f64>) -> PyResult<f64> {
        let pred = self.predict(py, x)?;
        r2(&y, &pred).map_err(to_py_err)
    }

    /// scikit-learn's clone protocol: a new, unfitted estimator with the same parameters.
    fn __sklearn_clone__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let params = slf.borrow().get_params(slf.py(), false)?;
        slf.get_type().call((), Some(&params))
    }

    /// scikit-learn's tags protocol: what kind of estimator this is.
    fn __sklearn_tags__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        sklearn_tags(py, "regressor")
    }

    /// scikit-learn's fitted check: true once `fit` has succeeded.
    fn __sklearn_is_fitted__(&self) -> bool {
        self.fitted.is_some()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        repr_of("GradientBoostingRegressor", &self.get_params(py, false)?)
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

#[pyclass(name = "RandomForestRegressor", module = "tpt_ml_proven", dict)]
pub struct PyRandomForestRegressor {
    n_estimators: usize,
    max_depth: usize,
    min_samples_leaf: usize,
    min_samples_split: usize,
    max_leaf_nodes: Option<usize>,
    max_features: Option<usize>,
    seed: u64,
    fitted: Option<RandomForestRegressor>,
}

impl PyRandomForestRegressor {
    fn build(&self) -> RandomForestRegressor {
        {
            let mut m = RandomForestRegressor::new(self.n_estimators, self.max_depth)
                .with_min_samples_leaf(self.min_samples_leaf)
                .with_min_samples_split(self.min_samples_split)
                .with_max_leaf_nodes(self.max_leaf_nodes)
                .with_seed(self.seed);
            if let Some(f) = self.max_features {
                m = m.with_max_features(f);
            }
            m
        }
    }

    fn encode(&self) -> PyResult<Vec<u8>> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .to_bytes()
            .map_err(to_py_err)
    }

    fn decode(bytes: &[u8]) -> PyResult<Self> {
        let m = RandomForestRegressor::from_bytes(bytes).map_err(to_py_err)?;
        Ok(Self {
            n_estimators: m.n_estimators_setting(),
            max_depth: m.max_depth(),
            min_samples_leaf: m.min_samples_leaf(),
            min_samples_split: m.min_samples_split(),
            max_leaf_nodes: m.max_leaf_nodes(),
            max_features: m.max_features(),
            seed: m.seed(),
            fitted: Some(m),
        })
    }
}

#[pymethods]
impl PyRandomForestRegressor {
    /// Each feature's share of the total impurity reduction, averaged over the trees. Raises
    /// AttributeError before fit, and for a model loaded from a file written before importances
    /// were stored.
    #[getter]
    fn feature_importances_(&self) -> PyResult<Vec<f64>> {
        self.fitted
            .as_ref()
            .and_then(|m| m.feature_importances())
            .ok_or_else(|| {
                PyAttributeError::new_err("feature_importances_ is not available for this model")
            })
    }

    /// Number of features seen in `fit`. Raises AttributeError before fit, as scikit-learn does.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        (self.fitted.as_ref().and_then(|m| m.n_features()))
            .ok_or_else(|| PyAttributeError::new_err("n_features_in_ is not available before fit"))
    }

    /// Constructor parameters, as scikit-learn names them.
    #[new]
    #[pyo3(signature = (n_estimators=100, *, max_depth=10, min_samples_leaf=1, min_samples_split=2, max_leaf_nodes=None, max_features=None, seed=0))]
    fn new(
        n_estimators: usize,
        max_depth: usize,
        min_samples_leaf: usize,
        min_samples_split: usize,
        max_leaf_nodes: Option<usize>,
        max_features: Option<usize>,
        seed: u64,
    ) -> PyResult<Self> {
        Ok(Self {
            n_estimators,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            max_leaf_nodes,
            max_features,
            seed,
            fitted: None,
        })
    }

    /// The constructor parameters, by name.
    #[pyo3(signature = (deep=true))]
    #[allow(unused_variables)]
    fn get_params<'py>(&self, py: Python<'py>, deep: bool) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("n_estimators", self.n_estimators)?;
        d.set_item("max_depth", self.max_depth)?;
        d.set_item("min_samples_leaf", self.min_samples_leaf)?;
        d.set_item("min_samples_split", self.min_samples_split)?;
        d.set_item("max_leaf_nodes", self.max_leaf_nodes)?;
        d.set_item("max_features", self.max_features)?;
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
                    "max_depth" => this.max_depth = value.extract()?,
                    "min_samples_leaf" => this.min_samples_leaf = value.extract()?,
                    "min_samples_split" => this.min_samples_split = value.extract()?,
                    "max_leaf_nodes" => this.max_leaf_nodes = value.extract()?,
                    "max_features" => this.max_features = value.extract()?,
                    "seed" => this.seed = value.extract()?,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter {other:?} for estimator RandomForestRegressor"
                        )))
                    }
                }
            }
        }
        Ok(slf)
    }

    /// Fits to a 2-D array and a target sequence, and returns the estimator.
    fn fit<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: Vec<f64>,
    ) -> PyResult<Bound<'py, Self>> {
        let m = matrix_from(py, x)?;
        let (r, c) = m.shape();
        let data = to_vec::<f64>(&m);
        if y.len() != r {
            return Err(PyValueError::new_err(
                "the target has a different length from the number of rows of X",
            ));
        }
        let mut model = slf.borrow().build();
        SupervisedEstimator::fit(
            &mut model,
            DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?,
            &y[..],
        )
        .map_err(to_py_err)?;
        slf.borrow_mut().fitted = Some(model);
        Ok(slf)
    }

    /// The predicted value (or label) for each row.
    fn predict(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let mm = matrix_from(py, x)?;
        let (r, c) = mm.shape();
        let data = to_vec::<f64>(&mm);
        let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
        Predictor::predict(m, view).map_err(to_py_err)
    }

    /// Fraction of rows whose predicted label equals the label in `y` (classifiers) or the R²
    /// of the predictions (regressors).
    fn score(&self, py: Python<'_>, x: &Bound<'_, PyAny>, y: Vec<f64>) -> PyResult<f64> {
        let pred = self.predict(py, x)?;
        r2(&y, &pred).map_err(to_py_err)
    }

    /// scikit-learn's clone protocol: a new, unfitted estimator with the same parameters.
    fn __sklearn_clone__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let params = slf.borrow().get_params(slf.py(), false)?;
        slf.get_type().call((), Some(&params))
    }

    /// scikit-learn's tags protocol: what kind of estimator this is.
    fn __sklearn_tags__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        sklearn_tags(py, "regressor")
    }

    /// scikit-learn's fitted check: true once `fit` has succeeded.
    fn __sklearn_is_fitted__(&self) -> bool {
        self.fitted.is_some()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        repr_of("RandomForestRegressor", &self.get_params(py, false)?)
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

#[pyclass(name = "RandomForestClassifier", module = "tpt_ml_proven", dict)]
pub struct PyRandomForestClassifier {
    n_estimators: usize,
    max_depth: usize,
    min_samples_leaf: usize,
    min_samples_split: usize,
    max_leaf_nodes: Option<usize>,
    criterion: String,
    max_features: Option<usize>,
    seed: u64,
    fitted: Option<RandomForestClassifier>,
}

impl PyRandomForestClassifier {
    fn build(&self) -> RandomForestClassifier {
        {
            let mut m = RandomForestClassifier::new(self.n_estimators, self.max_depth)
                .with_min_samples_leaf(self.min_samples_leaf)
                .with_min_samples_split(self.min_samples_split)
                .with_max_leaf_nodes(self.max_leaf_nodes)
                .with_criterion(parse_criterion(&self.criterion).expect("validated"))
                .with_seed(self.seed);
            if let Some(f) = self.max_features {
                m = m.with_max_features(f);
            }
            m
        }
    }

    fn encode(&self) -> PyResult<Vec<u8>> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .to_bytes()
            .map_err(to_py_err)
    }

    fn decode(bytes: &[u8]) -> PyResult<Self> {
        let m = RandomForestClassifier::from_bytes(bytes).map_err(to_py_err)?;
        Ok(Self {
            n_estimators: m.n_estimators_setting(),
            max_depth: m.max_depth(),
            min_samples_leaf: m.min_samples_leaf(),
            min_samples_split: m.min_samples_split(),
            max_leaf_nodes: m.max_leaf_nodes(),
            criterion: criterion_name(m.criterion()).to_string(),
            max_features: m.max_features(),
            seed: m.seed(),
            fitted: Some(m),
        })
    }
}

#[pymethods]
impl PyRandomForestClassifier {
    /// Number of features seen in `fit`. Raises AttributeError before fit, as scikit-learn does.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        (self.fitted.as_ref().and_then(|m| m.n_features()))
            .ok_or_else(|| PyAttributeError::new_err("n_features_in_ is not available before fit"))
    }

    /// Each feature's share of the total impurity reduction, averaged over the trees. Raises
    /// AttributeError before fit, and for a model loaded from a file written before importances
    /// were stored.
    #[getter]
    fn feature_importances_(&self) -> PyResult<Vec<f64>> {
        self.fitted
            .as_ref()
            .and_then(|m| m.feature_importances())
            .ok_or_else(|| {
                PyAttributeError::new_err("feature_importances_ is not available for this model")
            })
    }

    /// Constructor parameters, as scikit-learn names them.
    #[new]
    #[pyo3(signature = (n_estimators=100, *, max_depth=10, min_samples_leaf=1, min_samples_split=2, max_leaf_nodes=None, criterion="gini", max_features=None, seed=0))]
    fn new(
        n_estimators: usize,
        max_depth: usize,
        min_samples_leaf: usize,
        min_samples_split: usize,
        max_leaf_nodes: Option<usize>,
        criterion: &str,
        max_features: Option<usize>,
        seed: u64,
    ) -> PyResult<Self> {
        parse_criterion(criterion)?;
        Ok(Self {
            n_estimators,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            max_leaf_nodes,
            criterion: criterion.to_string(),
            max_features,
            seed,
            fitted: None,
        })
    }

    /// The constructor parameters, by name.
    #[pyo3(signature = (deep=true))]
    #[allow(unused_variables)]
    fn get_params<'py>(&self, py: Python<'py>, deep: bool) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("n_estimators", self.n_estimators)?;
        d.set_item("max_depth", self.max_depth)?;
        d.set_item("min_samples_leaf", self.min_samples_leaf)?;
        d.set_item("min_samples_split", self.min_samples_split)?;
        d.set_item("max_leaf_nodes", self.max_leaf_nodes)?;
        d.set_item("criterion", self.criterion.as_str())?;
        d.set_item("max_features", self.max_features)?;
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
                    "max_depth" => this.max_depth = value.extract()?,
                    "min_samples_leaf" => this.min_samples_leaf = value.extract()?,
                    "min_samples_split" => this.min_samples_split = value.extract()?,
                    "max_leaf_nodes" => this.max_leaf_nodes = value.extract()?,
                    "criterion" => this.criterion = value.extract()?,
                    "max_features" => this.max_features = value.extract()?,
                    "seed" => this.seed = value.extract()?,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter {other:?} for estimator RandomForestClassifier"
                        )))
                    }
                }
            }
        }
        Ok(slf)
    }

    /// Fits to a 2-D array and a target sequence, and returns the estimator.
    fn fit<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: Vec<f64>,
    ) -> PyResult<Bound<'py, Self>> {
        let m = matrix_from(py, x)?;
        let (r, c) = m.shape();
        let data = to_vec::<f64>(&m);
        if y.len() != r {
            return Err(PyValueError::new_err(
                "the target has a different length from the number of rows of X",
            ));
        }
        let mut model = slf.borrow().build();
        SupervisedEstimator::fit(
            &mut model,
            DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?,
            &y[..],
        )
        .map_err(to_py_err)?;
        slf.borrow_mut().fitted = Some(model);
        Ok(slf)
    }

    /// The predicted value (or label) for each row.
    fn predict(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let mm = matrix_from(py, x)?;
        let (r, c) = mm.shape();
        let data = to_vec::<f64>(&mm);
        let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
        Predictor::predict(m, view).map_err(to_py_err)
    }

    /// Fraction of rows whose predicted label equals the label in `y` (classifiers) or the R²
    /// of the predictions (regressors).
    fn score(&self, py: Python<'_>, x: &Bound<'_, PyAny>, y: Vec<f64>) -> PyResult<f64> {
        let pred = self.predict(py, x)?;
        accuracy(&y, &pred).map_err(to_py_err)
    }

    /// scikit-learn's clone protocol: a new, unfitted estimator with the same parameters.
    fn __sklearn_clone__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let params = slf.borrow().get_params(slf.py(), false)?;
        slf.get_type().call((), Some(&params))
    }

    /// scikit-learn's tags protocol: what kind of estimator this is.
    fn __sklearn_tags__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        sklearn_tags(py, "classifier")
    }

    /// scikit-learn's fitted check: true once `fit` has succeeded.
    fn __sklearn_is_fitted__(&self) -> bool {
        self.fitted.is_some()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        repr_of("RandomForestClassifier", &self.get_params(py, false)?)
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

    /// The distinct class labels seen in training, ascending.
    #[getter]
    fn classes_(&self) -> PyResult<Vec<f64>> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        m.classes().map(|c| c.to_vec()).ok_or_else(not_fitted)
    }
}
