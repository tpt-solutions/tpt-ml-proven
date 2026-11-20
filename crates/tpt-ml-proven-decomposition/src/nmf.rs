// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Non-negative matrix factorisation `X ≈ W·H` with `W, H ≥ 0`, minimising the Frobenius
//! error `‖X − WH‖²_F`, fitted by Lee–Seung multiplicative updates.
//!
//! The update rules and the stopping rule are those of scikit-learn's `solver="mu"` with
//! `beta_loss="frobenius"`, so the same starting matrices give the same iterates. Computation
//! is in `f64`.

use tpt_ml_proven_core::error::{Error, ShapeError};
use tpt_ml_proven_core::{
    ensures, requires, ArrayView, Dense, Float, Transformer, UnsupervisedEstimator,
};

/// Smallest value used in denominators, as in scikit-learn (`np.finfo(float).eps`).
const EPSILON: f64 = f64::EPSILON;

/// Non-negative matrix factorisation.
#[derive(Debug, Clone, PartialEq)]
pub struct Nmf<F: Float> {
    n_components: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    init: Option<(Vec<f64>, Vec<f64>)>,
    fitted: Option<Fitted<F>>,
    marker: core::marker::PhantomData<F>,
}

#[derive(Debug, Clone, PartialEq)]
struct Fitted<F: Float> {
    /// `H`, row-major `n_components × cols`, in the f64 the solver works in.
    components: Vec<f64>,
    /// The same `H` in the model's float type, for `components()`.
    factors: Dense<F>,
    /// `W` for the training data, row-major `rows × n_components`.
    w: Vec<f64>,
    cols: usize,
    n_iter: usize,
    reconstruction_error: f64,
}

impl<F: Float> Nmf<F> {
    /// An unfitted model with `n_components` factors. Defaults: 200 iterations,
    /// tolerance `1e-4`, seed 0 for the random start.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 200,
            tol: 1e-4,
            seed: 0,
            init: None,
            fitted: None,
            marker: core::marker::PhantomData,
        }
    }

    /// Sets the iteration limit (default 200).
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the relative error-decrease tolerance (default `1e-4`). Zero runs `max_iter`
    /// iterations exactly.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the seed for the random starting matrices (default 0).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Starts from the given `W` (`rows × k`, row-major) and `H` (`k × cols`, row-major)
    /// instead of random matrices. Both must be non-negative and have the right shape.
    pub fn with_init(mut self, w: Vec<f64>, h: Vec<f64>) -> Self {
        self.init = Some((w, h));
        self
    }

    /// Learned factors `H` (`n_components × n_features`), or `None` before `fit`.
    pub fn components(&self) -> Option<&Dense<F>> {
        self.fitted.as_ref().map(|f| &f.factors)
    }

    /// Multiplicative-update iterations used by the last fit, or `None` before `fit`.
    pub fn n_iter(&self) -> Option<usize> {
        self.fitted.as_ref().map(|f| f.n_iter)
    }

    /// Frobenius norm `‖X − WH‖_F` at the end of the last fit, or `None` before `fit`.
    pub fn reconstruction_error(&self) -> Option<f64> {
        self.fitted.as_ref().map(|f| f.reconstruction_error)
    }
}

/// Row-major matrix product `a (m×k) · b (k×n)`.
fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p];
            for j in 0..n {
                out[i * n + j] += aip * b[p * n + j];
            }
        }
    }
    out
}

/// Row-major `aᵀ · b`, with `a` of shape `m × k` and `b` of shape `m × n`; result `k × n`.
fn matmul_at_b(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; k * n];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p];
            for j in 0..n {
                out[p * n + j] += aip * b[i * n + j];
            }
        }
    }
    out
}

/// Row-major `a · bᵀ`, with `a` of shape `m × k` and `b` of shape `n × k`; result `m × n`.
fn matmul_a_bt(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = (0..k).map(|p| a[i * k + p] * b[j * k + p]).sum();
        }
    }
    out
}

/// Multiplicative update `M ← M ⊙ numerator ⊘ max(denominator, ε)`, elementwise.
fn multiply_update(m: &mut [f64], numerator: &[f64], denominator: &[f64]) {
    for ((v, &num), &den) in m.iter_mut().zip(numerator).zip(denominator) {
        let den = if den == 0.0 { EPSILON } else { den };
        *v *= num / den;
    }
}

/// `‖X − WH‖_F`.
fn frobenius_error(x: &[f64], w: &[f64], h: &[f64], rows: usize, k: usize, cols: usize) -> f64 {
    let wh = matmul(w, h, rows, k, cols);
    x.iter()
        .zip(&wh)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

/// SplitMix64 step, mapped to `[0, 1)`. A small private copy: this crate has no dependencies.
fn next_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// The non-negative data matrix: dense, or as its non-zero entries `(row, col, value)`. The
/// dense variant calls the dense products; the sparse one only visits the non-zeros, so a sparse
/// input is never copied into a dense buffer.
enum Data {
    Dense(Vec<f64>),
    Sparse(Vec<(usize, usize, f64)>),
}

impl Data {
    /// The mean of all `rows × cols` entries, zeros included.
    fn mean(&self, rows: usize, cols: usize) -> f64 {
        let total: f64 = match self {
            Data::Dense(d) => d.iter().sum(),
            Data::Sparse(nz) => nz.iter().map(|&(_, _, v)| v).sum(),
        };
        total / (rows * cols).max(1) as f64
    }

    /// `X · Hᵀ`, with `H` of shape `k × cols`: `rows × k`.
    fn xht(&self, h: &[f64], rows: usize, k: usize, cols: usize) -> Vec<f64> {
        match self {
            Data::Dense(d) => matmul_a_bt(d, h, rows, cols, k),
            Data::Sparse(nz) => {
                let mut out = vec![0.0_f64; rows * k];
                for &(r, c, v) in nz {
                    for p in 0..k {
                        out[r * k + p] += v * h[p * cols + c];
                    }
                }
                out
            }
        }
    }

    /// `Wᵀ · X`, with `W` of shape `rows × k`: `k × cols`.
    fn wtx(&self, w: &[f64], rows: usize, k: usize, cols: usize) -> Vec<f64> {
        match self {
            Data::Dense(d) => matmul_at_b(w, d, rows, k, cols),
            Data::Sparse(nz) => {
                let mut out = vec![0.0_f64; k * cols];
                for &(r, c, v) in nz {
                    for p in 0..k {
                        out[p * cols + c] += w[r * k + p] * v;
                    }
                }
                out
            }
        }
    }

    /// `‖X − W·H‖_F`. The sparse form uses `‖X − WH‖² = Σ_nz (x² − 2·x·(WH)) + ‖WH‖²`, where the
    /// last term is `Σ (WᵀW ⊙ HHᵀ)`, so no dense product is formed.
    fn error(&self, w: &[f64], h: &[f64], rows: usize, k: usize, cols: usize) -> f64 {
        match self {
            Data::Dense(d) => frobenius_error(d, w, h, rows, k, cols),
            Data::Sparse(nz) => {
                let wtw = matmul_at_b(w, w, rows, k, k);
                let hht = matmul_a_bt(h, h, k, cols, k);
                let wh_norm2: f64 = wtw.iter().zip(&hht).map(|(a, b)| a * b).sum();
                let cross: f64 = nz
                    .iter()
                    .map(|&(r, c, v)| {
                        let wh: f64 = (0..k).map(|p| w[r * k + p] * h[p * cols + c]).sum();
                        v * v - 2.0 * v * wh
                    })
                    .sum();
                (cross + wh_norm2).max(0.0).sqrt()
            }
        }
    }
}

/// Reads the data matrix, checking that every entry is non-negative. A sparse input is read from
/// its non-zeros, and a dense one entry by entry.
fn gather<F: Float, X: ArrayView<Elem = F>>(
    x: &X,
    rows: usize,
    cols: usize,
) -> Result<Data, Error> {
    let negative = || Error::InvalidParameter {
        name: "X",
        reason: "must be non-negative",
    };
    if x.is_sparse() {
        let mut nz = Vec::new();
        for r in 0..rows {
            x.for_each_nonzero(r, |c, v| nz.push((r, c, v.to_f64())));
        }
        if nz.iter().any(|&(_, _, v)| v < 0.0) {
            return Err(negative());
        }
        return Ok(Data::Sparse(nz));
    }
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let v = x.get(r, c).to_f64();
            if v < 0.0 {
                return Err(negative());
            }
            data.push(v);
        }
    }
    Ok(Data::Dense(data))
}

impl<F: Float> UnsupervisedEstimator for Nmf<F> {
    type Float = F;
    type FitError = Error;

    fn fit<X>(&mut self, x: X) -> Result<(), Error>
    where
        X: ArrayView<Elem = F>,
    {
        self.fitted = None;
        let (rows, cols) = (x.rows(), x.cols());
        if rows == 0 || cols == 0 {
            return Err(Error::EmptyInput);
        }
        let k = self.n_components;
        if k == 0 {
            return Err(Error::InvalidParameter {
                name: "n_components",
                reason: "must be at least 1",
            });
        }
        if !x.all_finite() {
            return Err(Error::NonFinite { what: "X" });
        }
        let data = gather(&x, rows, cols)?;

        let (mut w, mut h) = match &self.init {
            Some((w, h)) => {
                if w.len() != rows * k || h.len() != k * cols {
                    return Err(Error::InvalidParameter {
                        name: "init",
                        reason: "W must be rows × n_components and H must be n_components × cols",
                    });
                }
                if w.iter().chain(h).any(|&v| !(v >= 0.0 && v.is_finite())) {
                    return Err(Error::InvalidParameter {
                        name: "init",
                        reason: "initial factors must be finite and non-negative",
                    });
                }
                (w.clone(), h.clone())
            }
            None => {
                // Scale so that W·H has about the mean of X, then draw uniform entries.
                let mean = data.mean(rows, cols);
                let scale = (mean.max(EPSILON) / k as f64).sqrt();
                let mut state = self.seed;
                let w: Vec<f64> = (0..rows * k)
                    .map(|_| scale * next_unit(&mut state))
                    .collect();
                let h: Vec<f64> = (0..k * cols)
                    .map(|_| scale * next_unit(&mut state))
                    .collect();
                (w, h)
            }
        };

        requires!(rows >= 1, "nmf.rows_positive");
        requires!(k >= 1, "nmf.components_positive");
        let error_at_init = data.error(&w, &h, rows, k, cols);
        let mut previous_error = error_at_init;
        let mut n_iter = 0;
        for iter in 1..=self.max_iter {
            n_iter = iter;
            // W ← W ⊙ (X Hᵀ) ⊘ (W H Hᵀ)
            let xht = data.xht(&h, rows, k, cols);
            let hht = matmul_a_bt(&h, &h, k, cols, k);
            let whht = matmul(&w, &hht, rows, k, k);
            multiply_update(&mut w, &xht, &whht);

            // H ← H ⊙ (Wᵀ X) ⊘ (Wᵀ W H)
            let wtx = data.wtx(&w, rows, k, cols);
            let wtw = matmul_at_b(&w, &w, rows, k, k);
            let wtwh = matmul(&wtw, &h, k, k, cols);
            multiply_update(&mut h, &wtx, &wtwh);

            // Convergence is tested every 10 iterations, as in scikit-learn.
            if self.tol > 0.0 && iter % 10 == 0 {
                let error = data.error(&w, &h, rows, k, cols);
                let decrease = (previous_error - error) / error_at_init;
                previous_error = error;
                if decrease < self.tol {
                    break;
                }
            }
        }

        let reconstruction_error = data.error(&w, &h, rows, k, cols);
        // Multiplicative updates keep the factors non-negative and do not increase the error.
        ensures!(
            w.iter().chain(&h).all(|&v| v >= 0.0),
            "nmf.factors_nonnegative"
        );
        ensures!(
            reconstruction_error <= error_at_init * (1.0 + 1e-9),
            "nmf.error_not_increased"
        );
        let factors = Dense::from_vec(h.iter().map(|v| F::from_f64(*v)).collect(), k, cols)?;
        self.fitted = Some(Fitted {
            components: h,
            factors,
            w,
            cols,
            n_iter,
            reconstruction_error,
        });
        Ok(())
    }
}

impl<F: Float> Transformer for Nmf<F> {
    type Float = F;
    type Output = Dense<F>;

    /// The factor `W` for new data, with `H` held fixed: `rows × n_components`. Computed by
    /// multiplicative updates on `W` alone, for as many iterations as the fit used.
    fn transform<X>(&self, x: X) -> Result<Dense<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        let fitted = self.fitted.as_ref().ok_or(Error::NotFitted)?;
        let (rows, cols) = (x.rows(), x.cols());
        if cols != fitted.cols {
            return Err(Error::Shape(ShapeError {
                rows,
                cols,
                reason: "feature count differs from the fitted model",
            }));
        }
        if !x.all_finite() {
            return Err(Error::NonFinite { what: "X" });
        }
        let k = self.n_components;
        let h = &fitted.components;
        let data = gather(&x, rows, cols)?;
        // Start from the mean-scaled constant, then update W alone for the same budget.
        let mean = data.mean(rows, cols);
        let mut w = vec![(mean.max(EPSILON) / k as f64).sqrt(); rows * k];
        let hht = matmul_a_bt(h, h, k, cols, k);
        for _ in 0..fitted.n_iter.max(1) {
            let xht = data.xht(h, rows, k, cols);
            let whht = matmul(&w, &hht, rows, k, k);
            multiply_update(&mut w, &xht, &whht);
        }
        let out: Vec<F> = w.into_iter().map(F::from_f64).collect();
        Ok(Dense::from_vec(out, rows, k)?)
    }

    /// The training factor `W` from the fit itself (not recomputed by `transform`).
    fn fit_transform<X>(&mut self, x: X) -> Result<Dense<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        UnsupervisedEstimator::fit(self, &x)?;
        let fitted = self.fitted.as_ref().ok_or(Error::NotFitted)?;
        let k = self.n_components;
        let rows = fitted.w.len() / k;
        let out: Vec<F> = fitted.w.iter().map(|&v| F::from_f64(v)).collect();
        Ok(Dense::from_vec(out, rows, k)?)
    }
}

// ---------------------------------------------------------------- model files

use tpt_ml_proven_core::persist::{kind, Reader, Writer};

fn invalid(reason: &'static str) -> Error {
    Error::Format { reason }
}

impl<F: Float> Nmf<F> {
    /// Serialises the model (settings and fitted factors) to a model file.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut w = Writer::for_float::<F>(kind::NMF);
        w.usize(self.n_components);
        w.usize(self.max_iter);
        w.f64(self.tol);
        w.u64(self.seed);
        match &self.init {
            None => w.bool(false),
            Some((w_init, h_init)) => {
                w.bool(true);
                w.f64s(w_init);
                w.f64s(h_init);
            }
        }
        match &self.fitted {
            None => w.bool(false),
            Some(f) => {
                w.bool(true);
                w.usize(f.cols);
                w.f64s(&f.components);
                w.f64s(&f.w);
                w.usize(f.n_iter);
                w.f64(f.reconstruction_error);
            }
        }
        Ok(w.finish())
    }

    /// Loads a model written by [`Nmf::to_bytes`], checking the factor shapes and that every
    /// factor entry is non-negative.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(bytes, kind::NMF)?;
        let n_components = r.usize()?;
        let max_iter = r.usize()?;
        let tol = r.f64()?;
        let seed = r.u64()?;
        let init = if r.bool()? {
            Some((r.f64s()?, r.f64s()?))
        } else {
            None
        };
        let fitted = if r.bool()? {
            let cols = r.usize()?;
            let components = r.f64s()?;
            let w = r.f64s()?;
            let n_iter = r.usize()?;
            let reconstruction_error = r.f64()?;
            if n_components == 0 || cols == 0 {
                return Err(invalid("NMF has no components or no features"));
            }
            if components.len() != n_components * cols {
                return Err(invalid("NMF H does not match n_components x n_features"));
            }
            if w.len() % n_components != 0 {
                return Err(invalid("NMF W does not match n_components"));
            }
            if components.iter().chain(&w).any(|&v| v < 0.0) {
                return Err(invalid("NMF factors have a negative entry"));
            }
            let factors = Dense::from_vec(
                components.iter().map(|v| F::from_f64(*v)).collect(),
                n_components,
                cols,
            )?;
            Some(Fitted {
                components,
                factors,
                w,
                cols,
                n_iter,
                reconstruction_error,
            })
        } else {
            None
        };
        r.finish()?;
        Ok(Self {
            n_components,
            max_iter,
            tol,
            seed,
            init,
            fitted,
            marker: core::marker::PhantomData,
        })
    }
}

// ---------------------------------------------------------------- settings

impl<F: Float> Nmf<F> {
    /// The number of components the model was configured with.
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// The iteration limit the model was configured with.
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// The convergence tolerance the model was configured with.
    pub fn tol(&self) -> f64 {
        self.tol
    }

    /// The seed the model was configured with.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The starting factors `(W, H)` the model was configured with, if any.
    pub fn initial_factors(&self) -> Option<(Vec<f64>, Vec<f64>)> {
        self.init.clone()
    }
}

#[cfg(test)]
mod persist_tests {
    use super::*;
    use tpt_ml_proven_core::{DenseView, Transformer, UnsupervisedEstimator};

    fn data() -> Vec<f64> {
        (0..12).map(|i| ((i * 5) % 7) as f64 + 0.5).collect()
    }

    #[test]
    fn fitted_nmf_round_trips_and_transforms_identically() {
        let x = data();
        let mut m = Nmf::<f64>::new(2).with_seed(3);
        UnsupervisedEstimator::fit(&mut m, DenseView::new(&x, 4, 3).unwrap()).unwrap();
        let loaded = Nmf::<f64>::from_bytes(&m.to_bytes().unwrap()).unwrap();
        assert_eq!(loaded, m);
        let a = Transformer::transform(&m, DenseView::new(&x, 4, 3).unwrap()).unwrap();
        let b = Transformer::transform(&loaded, DenseView::new(&x, 4, 3).unwrap()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn unfitted_nmf_round_trips_with_custom_init() {
        let m = Nmf::<f64>::new(1).with_init(vec![1.0, 2.0], vec![0.5, 0.25, 1.0]);
        assert_eq!(Nmf::<f64>::from_bytes(&m.to_bytes().unwrap()).unwrap(), m);
    }
}
