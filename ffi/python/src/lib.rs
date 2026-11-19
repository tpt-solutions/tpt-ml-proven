// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Python bindings for tpt-ml-proven: `KMeans` and `PCA`.
//!
//! Inputs are C-contiguous 2-D `float32` or `float64` NumPy arrays. A matching dtype
//! is read in place, with no copy. `fit` releases the GIL. Outputs are plain Python
//! lists, so the bindings have no NumPy dependency at run time.
//!
//! Zero-copy reads use NumPy's `__array_interface__`, not the buffer protocol. The
//! buffer protocol is not part of the stable ABI before Python 3.11, and the wheels
//! are built `abi3` for Python 3.9 and later. The array is kept alive by the binding
//! while it is used.
//!
//! The array must not be modified while `fit` runs. The memory is read without a lock,
//! so another thread writing to it would race with the fit.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};

use tpt_ml_proven_clustering::{Init, KMeans};
use tpt_ml_proven_core::error::Error;
use tpt_ml_proven_core::{
    ArrayView, Dense, DenseView, Float, Predictor, Transformer, UnsupervisedEstimator,
};
use tpt_ml_proven_decomposition::Pca;

mod models;

pub use models::{
    PyDbscan, PyElasticNet, PyHdbscan, PyLasso, PyLogisticRegression, PyNmf, PyRidge,
    PyStumpForest, PyTruncatedSvd,
};

/// Dtype of a model file: 1 for f32, 2 for f64. Version 1 files are f64.
pub(crate) fn peek_dtype(bytes: &[u8]) -> PyResult<u32> {
    if bytes.len() < 20 || bytes[..8] != *b"TPTMLMDL" {
        return Err(PyValueError::new_err("not a tpt-ml-proven model file"));
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes"));
    if version < 2 {
        return Ok(2);
    }
    Ok(u32::from_le_bytes(bytes[16..20].try_into().expect("4 bytes")))
}

fn to_py_err(e: Error) -> PyErr {
    match e {
        Error::NotFitted => PyRuntimeError::new_err(e.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn not_fitted() -> PyErr {
    PyRuntimeError::new_err("this estimator is not fitted yet; call fit first")
}

fn shape_error() -> PyErr {
    PyValueError::new_err(
        "expected a 2-D C-contiguous float32 or float64 array (for example a NumPy array)",
    )
}

/// A validated, C-contiguous 2-D array borrowed from Python.
///
/// `_owner` keeps the Python object alive, so `ptr` stays valid for the lifetime of
/// this value.
pub struct Borrowed {
    _owner: Py<PyAny>,
    ptr: usize,
    rows: usize,
    cols: usize,
}

/// Element type of a borrowed array.
pub enum Matrix {
    F32(Borrowed),
    F64(Borrowed),
}

impl Matrix {
    fn shape(&self) -> (usize, usize) {
        match self {
            Matrix::F32(b) | Matrix::F64(b) => (b.rows, b.cols),
        }
    }
}

/// Reads `__array_interface__` and checks that the array is 2-D, C-contiguous, and of
/// dtype `<f4` or `<f8`.
fn matrix_from(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Matrix> {
    let iface = obj
        .getattr("__array_interface__")
        .map_err(|_| shape_error())?;
    let dict = iface.cast::<PyDict>().map_err(|_| shape_error())?;

    let typestr: String = dict
        .get_item("typestr")?
        .ok_or_else(shape_error)?
        .extract()?;
    let (is_f64, itemsize) = match typestr.as_str() {
        "<f8" | "=f8" | "|f8" => (true, 8_usize),
        "<f4" | "=f4" | "|f4" => (false, 4_usize),
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported dtype {typestr:?}; expected float32 or float64"
            )))
        }
    };

    let shape: Vec<usize> = dict.get_item("shape")?.ok_or_else(shape_error)?.extract()?;
    if shape.len() != 2 {
        return Err(shape_error());
    }
    let (rows, cols) = (shape[0], shape[1]);

    // strides is None for C-contiguous arrays. Otherwise it must match C order.
    if let Some(strides) = dict.get_item("strides")? {
        if !strides.is_none() {
            let s: Vec<isize> = strides.extract()?;
            let expect = [(cols * itemsize) as isize, itemsize as isize];
            if s.len() != 2 || s[..] != expect[..] {
                return Err(PyValueError::new_err(
                    "array must be C-contiguous; call np.ascontiguousarray first",
                ));
            }
        }
    }

    let data = dict.get_item("data")?.ok_or_else(shape_error)?;
    let data = data.cast::<PyTuple>().map_err(|_| shape_error())?;
    let ptr: usize = data.get_item(0)?.extract()?;

    let b = Borrowed {
        _owner: obj.clone().unbind(),
        ptr,
        rows,
        cols,
    };
    let _ = py;
    Ok(if is_f64 {
        Matrix::F64(b)
    } else {
        Matrix::F32(b)
    })
}

impl Borrowed {
    fn len(&self) -> usize {
        self.rows * self.cols
    }

    /// The elements as `f64`. The caller must have checked the dtype.
    fn as_f64(&self) -> &[f64] {
        if self.len() == 0 {
            return &[];
        }
        // SAFETY: the dtype was checked as `<f8` and the array is C-contiguous with
        // `rows * cols` elements. `_owner` keeps the array alive.
        unsafe { std::slice::from_raw_parts(self.ptr as *const f64, self.len()) }
    }

    /// The elements as `f32`. The caller must have checked the dtype.
    fn as_f32(&self) -> &[f32] {
        if self.len() == 0 {
            return &[];
        }
        // SAFETY: as in `as_f64`, for `<f4`.
        unsafe { std::slice::from_raw_parts(self.ptr as *const f32, self.len()) }
    }
}

/// Copies a matrix into element type `F`. Only used when the input dtype differs from
/// the fitted model's.
fn to_vec<F: Float>(m: &Matrix) -> Vec<F> {
    match m {
        Matrix::F64(b) => b.as_f64().iter().map(|&v| F::from_f64(v)).collect(),
        Matrix::F32(b) => b
            .as_f32()
            .iter()
            .map(|&v| F::from_f64(f64::from(v)))
            .collect(),
    }
}

fn rows_of<F: Float>(d: &Dense<F>) -> Vec<Vec<f64>> {
    (0..d.rows())
        .map(|r| (0..d.cols()).map(|c| d.get(r, c).to_f64()).collect())
        .collect()
}

// ---------------------------------------------------------------- KMeans

#[derive(Clone)]
struct KMeansParams {
    n_clusters: usize,
    max_iter: usize,
    tol: f64,
    init: Init,
    seed: u64,
    threads: usize,
}

impl KMeansParams {
    fn build<F: Float>(&self) -> KMeans<F> {
        KMeans::<F>::new(self.n_clusters)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_init(self.init)
            .with_seed(self.seed)
            .with_threads(self.threads)
    }
}

enum KState {
    F32(KMeans<f32>),
    F64(KMeans<f64>),
}

/// KMeans clustering by Lloyd's algorithm.
///
/// `init` is `"k-means++"` (default) or `"random"`. `threads` sets the worker count for
/// the assignment and update steps. Results do not depend on it.
#[pyclass(name = "KMeans", module = "tpt_ml_proven")]
pub struct PyKMeans {
    params: KMeansParams,
    fitted: Option<KState>,
}

impl PyKMeans {
    /// Fits to a 2-D `float32` or `float64` array. Releases the GIL while fitting.
    fn fit_impl(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<()> {
        self.fitted = None;
        let m = matrix_from(py, x)?;
        let params = self.params.clone();
        let state = match &m {
            Matrix::F64(b) => {
                let (r, c) = (b.rows, b.cols);
                let data = b.as_f64();
                let km = py.detach(move || -> Result<KMeans<f64>, Error> {
                    let mut km = params.build::<f64>();
                    UnsupervisedEstimator::fit(&mut km, DenseView::new(data, r, c)?)?;
                    Ok(km)
                });
                KState::F64(km.map_err(to_py_err)?)
            }
            Matrix::F32(b) => {
                let (r, c) = (b.rows, b.cols);
                let data = b.as_f32();
                let km = py.detach(move || -> Result<KMeans<f32>, Error> {
                    let mut km = params.build::<f32>();
                    UnsupervisedEstimator::fit(&mut km, DenseView::new(data, r, c)?)?;
                    Ok(km)
                });
                KState::F32(km.map_err(to_py_err)?)
            }
        };
        self.fitted = Some(state);
        Ok(())
    }
}

impl PyKMeans {
    fn encode(&self) -> PyResult<Vec<u8>> {
        match self.fitted.as_ref().ok_or_else(not_fitted)? {
            KState::F32(m) => m.to_bytes().map_err(to_py_err),
            KState::F64(m) => m.to_bytes().map_err(to_py_err),
        }
    }

    fn decode(bytes: &[u8]) -> PyResult<Self> {
        fn params<F: Float>(m: &KMeans<F>) -> KMeansParams {
            KMeansParams {
                n_clusters: m.n_clusters(),
                max_iter: m.max_iter(),
                tol: m.tol(),
                init: m.init(),
                seed: m.seed(),
                threads: m.threads(),
            }
        }
        if peek_dtype(bytes)? == 1 {
            let m = KMeans::<f32>::from_bytes(bytes).map_err(to_py_err)?;
            Ok(Self {
                params: params(&m),
                fitted: Some(KState::F32(m)),
            })
        } else {
            let m = KMeans::<f64>::from_bytes(bytes).map_err(to_py_err)?;
            Ok(Self {
                params: params(&m),
                fitted: Some(KState::F64(m)),
            })
        }
    }
}

#[pymethods]
impl PyKMeans {
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
    #[pyo3(signature = (n_clusters, *, max_iter=300, tol=1e-4, init="k-means++", seed=0, threads=1))]
    fn new(
        n_clusters: usize,
        max_iter: usize,
        tol: f64,
        init: &str,
        seed: u64,
        threads: usize,
    ) -> PyResult<Self> {
        let init = match init {
            "k-means++" => Init::KMeansPlusPlus,
            "random" => Init::Random,
            other => {
                return Err(PyValueError::new_err(format!(
                    "init must be 'k-means++' or 'random', not {other:?}"
                )))
            }
        };
        Ok(Self {
            params: KMeansParams {
                n_clusters,
                max_iter,
                tol,
                init,
                seed,
                threads,
            },
            fitted: None,
        })
    }

    /// Fits to a 2-D `float32` or `float64` array and returns the estimator, so calls
    /// can be chained. Releases the GIL while fitting.
    fn fit<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.borrow_mut().fit_impl(py, x)?;
        Ok(slf)
    }

    /// Nearest centre for each row, as a list of cluster indices.
    fn predict(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
        let state = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let m = matrix_from(py, x)?;
        let (r, c) = m.shape();
        match state {
            KState::F64(km) => {
                let data: Vec<f64> = to_vec(&m);
                let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
                km.predict(view).map_err(to_py_err)
            }
            KState::F32(km) => {
                let data: Vec<f32> = to_vec(&m);
                let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
                km.predict(view).map_err(to_py_err)
            }
        }
    }

    /// Squared distance from each row to each centre, as a list of rows.
    fn transform(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        let state = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let m = matrix_from(py, x)?;
        let (r, c) = m.shape();
        match state {
            KState::F64(km) => {
                let data: Vec<f64> = to_vec(&m);
                let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
                Ok(rows_of(&km.transform(view).map_err(to_py_err)?))
            }
            KState::F32(km) => {
                let data: Vec<f32> = to_vec(&m);
                let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
                Ok(rows_of(&km.transform(view).map_err(to_py_err)?))
            }
        }
    }

    /// Fits, then returns the squared distances to the centres.
    fn fit_transform(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        self.fit_impl(py, x)?;
        self.transform(py, x)
    }

    /// Cluster label of each training row.
    #[getter]
    fn labels_(&self) -> PyResult<Vec<usize>> {
        match self.fitted.as_ref().ok_or_else(not_fitted)? {
            KState::F64(km) => Ok(km.labels().unwrap_or(&[]).to_vec()),
            KState::F32(km) => Ok(km.labels().unwrap_or(&[]).to_vec()),
        }
    }

    /// Fitted centres, as a list of rows.
    #[getter]
    fn cluster_centers_(&self) -> PyResult<Vec<Vec<f64>>> {
        match self.fitted.as_ref().ok_or_else(not_fitted)? {
            KState::F64(km) => Ok(rows_of(km.cluster_centers().ok_or_else(not_fitted)?)),
            KState::F32(km) => Ok(rows_of(km.cluster_centers().ok_or_else(not_fitted)?)),
        }
    }

    /// Sum of squared distances from each training row to its centre.
    #[getter]
    fn inertia_(&self) -> PyResult<f64> {
        match self.fitted.as_ref().ok_or_else(not_fitted)? {
            KState::F64(km) => km.inertia().ok_or_else(not_fitted),
            KState::F32(km) => Ok(km.inertia().ok_or_else(not_fitted)?.to_f64()),
        }
    }

    /// Lloyd iterations performed.
    #[getter]
    fn n_iter_(&self) -> PyResult<usize> {
        match self.fitted.as_ref().ok_or_else(not_fitted)? {
            KState::F64(km) => km.n_iter().ok_or_else(not_fitted),
            KState::F32(km) => km.n_iter().ok_or_else(not_fitted),
        }
    }

    /// `True` if the assignments reached a fixed point.
    #[getter]
    fn converged_(&self) -> PyResult<bool> {
        match self.fitted.as_ref().ok_or_else(not_fitted)? {
            KState::F64(km) => km.converged().ok_or_else(not_fitted),
            KState::F32(km) => km.converged().ok_or_else(not_fitted),
        }
    }
}

// ---------------------------------------------------------------- PCA

enum PState {
    F32(Pca<f32>),
    F64(Pca<f64>),
}

/// Principal component analysis.
#[pyclass(name = "PCA", module = "tpt_ml_proven")]
pub struct PyPca {
    n_components: usize,
    fitted: Option<PState>,
}

impl PyPca {
    /// Fits to a 2-D `float32` or `float64` array. Releases the GIL while fitting.
    fn fit_impl(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<()> {
        self.fitted = None;
        let m = matrix_from(py, x)?;
        let k = self.n_components;
        let state = match &m {
            Matrix::F64(b) => {
                let (r, c) = (b.rows, b.cols);
                let data = b.as_f64();
                let p = py.detach(move || -> Result<Pca<f64>, Error> {
                    let mut p = Pca::<f64>::new(k);
                    UnsupervisedEstimator::fit(&mut p, DenseView::new(data, r, c)?)?;
                    Ok(p)
                });
                PState::F64(p.map_err(to_py_err)?)
            }
            Matrix::F32(b) => {
                let (r, c) = (b.rows, b.cols);
                let data = b.as_f32();
                let p = py.detach(move || -> Result<Pca<f32>, Error> {
                    let mut p = Pca::<f32>::new(k);
                    UnsupervisedEstimator::fit(&mut p, DenseView::new(data, r, c)?)?;
                    Ok(p)
                });
                PState::F32(p.map_err(to_py_err)?)
            }
        };
        self.fitted = Some(state);
        Ok(())
    }
}

impl PyPca {
    fn encode(&self) -> PyResult<Vec<u8>> {
        match self.fitted.as_ref().ok_or_else(not_fitted)? {
            PState::F32(m) => m.to_bytes().map_err(to_py_err),
            PState::F64(m) => m.to_bytes().map_err(to_py_err),
        }
    }

    fn decode(bytes: &[u8]) -> PyResult<Self> {
        if peek_dtype(bytes)? == 1 {
            let m = Pca::<f32>::from_bytes(bytes).map_err(to_py_err)?;
            Ok(Self {
                n_components: m.n_components(),
                fitted: Some(PState::F32(m)),
            })
        } else {
            let m = Pca::<f64>::from_bytes(bytes).map_err(to_py_err)?;
            Ok(Self {
                n_components: m.n_components(),
                fitted: Some(PState::F64(m)),
            })
        }
    }
}

#[pymethods]
impl PyPca {
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

    /// Fits to a 2-D `float32` or `float64` array and returns the estimator, so calls
    /// can be chained. Releases the GIL while fitting.
    fn fit<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.borrow_mut().fit_impl(py, x)?;
        Ok(slf)
    }

    /// Projects the rows onto the kept components.
    fn transform(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        let state = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let m = matrix_from(py, x)?;
        let (r, c) = m.shape();
        match state {
            PState::F64(p) => {
                let data: Vec<f64> = to_vec(&m);
                let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
                Ok(rows_of(&p.transform(view).map_err(to_py_err)?))
            }
            PState::F32(p) => {
                let data: Vec<f32> = to_vec(&m);
                let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
                Ok(rows_of(&p.transform(view).map_err(to_py_err)?))
            }
        }
    }

    /// Maps component scores back to feature space.
    fn inverse_transform(&self, py: Python<'_>, z: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        let state = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let m = matrix_from(py, z)?;
        let (r, c) = m.shape();
        match state {
            PState::F64(p) => {
                let data: Vec<f64> = to_vec(&m);
                let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
                Ok(rows_of(&p.inverse_transform(&view).map_err(to_py_err)?))
            }
            PState::F32(p) => {
                let data: Vec<f32> = to_vec(&m);
                let view = DenseView::new(&data, r, c).map_err(|e| to_py_err(e.into()))?;
                Ok(rows_of(&p.inverse_transform(&view).map_err(to_py_err)?))
            }
        }
    }

    /// Fits, then projects the same rows.
    fn fit_transform(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        self.fit_impl(py, x)?;
        self.transform(py, x)
    }

    /// Principal axes, one per row.
    #[getter]
    fn components_(&self) -> PyResult<Vec<Vec<f64>>> {
        match self.fitted.as_ref().ok_or_else(not_fitted)? {
            PState::F64(p) => Ok(rows_of(p.components().ok_or_else(not_fitted)?)),
            PState::F32(p) => Ok(rows_of(p.components().ok_or_else(not_fitted)?)),
        }
    }

    /// Per-feature mean removed before projection.
    #[getter]
    fn mean_(&self) -> PyResult<Vec<f64>> {
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
    #[getter]
    fn explained_variance_(&self) -> PyResult<Vec<f64>> {
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
    #[getter]
    fn explained_variance_ratio_(&self) -> PyResult<Vec<f64>> {
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

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKMeans>()?;
    m.add_class::<PyPca>()?;
    m.add_class::<PyDbscan>()?;
    m.add_class::<PyHdbscan>()?;
    m.add_class::<PyTruncatedSvd>()?;
    m.add_class::<PyNmf>()?;
    m.add_class::<PyRidge>()?;
    m.add_class::<PyLasso>()?;
    m.add_class::<PyElasticNet>()?;
    m.add_class::<PyLogisticRegression>()?;
    m.add_class::<PyStumpForest>()?;
    Ok(())
}
