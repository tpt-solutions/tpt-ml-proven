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
    fitted: Option<Fitted>,
    marker: core::marker::PhantomData<F>,
}

#[derive(Debug, Clone, PartialEq)]
struct Fitted {
    /// `H`, row-major `n_components × cols`.
    components: Vec<f64>,
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

    /// Learned factors `H` (`n_components × n_features`, row-major), or `None` before `fit`.
    pub fn components(&self) -> Option<&[f64]> {
        self.fitted.as_ref().map(|f| f.components.as_slice())
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
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let v = x.get(r, c).to_f64();
                if v < 0.0 {
                    return Err(Error::InvalidParameter {
                        name: "X",
                        reason: "must be non-negative",
                    });
                }
                data.push(v);
            }
        }

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
                let mean = data.iter().sum::<f64>() / data.len() as f64;
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
        let error_at_init = frobenius_error(&data, &w, &h, rows, k, cols);
        let mut previous_error = error_at_init;
        let mut n_iter = 0;
        for iter in 1..=self.max_iter {
            n_iter = iter;
            // W ← W ⊙ (X Hᵀ) ⊘ (W H Hᵀ)
            let xht = matmul_a_bt(&data, &h, rows, cols, k);
            let hht = matmul_a_bt(&h, &h, k, cols, k);
            let whht = matmul(&w, &hht, rows, k, k);
            multiply_update(&mut w, &xht, &whht);

            // H ← H ⊙ (Wᵀ X) ⊘ (Wᵀ W H)
            let wtx = matmul_at_b(&w, &data, rows, k, cols);
            let wtw = matmul_at_b(&w, &w, rows, k, k);
            let wtwh = matmul(&wtw, &h, k, k, cols);
            multiply_update(&mut h, &wtx, &wtwh);

            // Convergence is tested every 10 iterations, as in scikit-learn.
            if self.tol > 0.0 && iter % 10 == 0 {
                let error = frobenius_error(&data, &w, &h, rows, k, cols);
                let decrease = (previous_error - error) / error_at_init;
                previous_error = error;
                if decrease < self.tol {
                    break;
                }
            }
        }

        let reconstruction_error = frobenius_error(&data, &w, &h, rows, k, cols);
        // Multiplicative updates keep the factors non-negative and do not increase the error.
        ensures!(
            w.iter().chain(&h).all(|&v| v >= 0.0),
            "nmf.factors_nonnegative"
        );
        ensures!(
            reconstruction_error <= error_at_init * (1.0 + 1e-9),
            "nmf.error_not_increased"
        );
        self.fitted = Some(Fitted {
            components: h,
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
        let data: Vec<f64> = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (r, c)))
            .map(|(r, c)| x.get(r, c).to_f64())
            .collect();
        if data.iter().any(|&v| v < 0.0) {
            return Err(Error::InvalidParameter {
                name: "X",
                reason: "must be non-negative",
            });
        }
        // Start from the mean-scaled constant, then update W alone for the same budget.
        let mean = data.iter().sum::<f64>() / data.len().max(1) as f64;
        let mut w = vec![(mean.max(EPSILON) / k as f64).sqrt(); rows * k];
        let hht = matmul_a_bt(h, h, k, cols, k);
        for _ in 0..fitted.n_iter.max(1) {
            let xht = matmul_a_bt(&data, h, rows, cols, k);
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
