// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Binary logistic regression, fitted by damped Newton iterations.
//!
//! The objective is the one scikit-learn minimises with `C`, for labels in `{0, 1}`:
//!
//! `½‖w‖² + C · Σᵢ [log(1 + e^{zᵢ}) − yᵢ zᵢ]`, with `zᵢ = wᵀxᵢ + b`.
//!
//! The intercept `b` is not penalised. Computation is in `f64`.

use tpt_ml_proven_core::error::{Error, ShapeError};
use tpt_ml_proven_core::{
    ensures, requires, ArrayView, Float, Predictor, SupervisedEstimator, TargetView,
};

use crate::cholesky_solve;

/// Numerically stable `log(1 + e^z)`.
fn softplus(z: f64) -> f64 {
    z.max(0.0) + (-z.abs()).exp().ln_1p()
}

/// Numerically stable logistic function.
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Binary logistic regression with an L2 penalty controlled by `C` (larger `C` means
/// weaker regularisation), as in scikit-learn.
#[derive(Debug, Clone, PartialEq)]
pub struct LogisticRegression<F: Float> {
    c: f64,
    max_iter: usize,
    tol: f64,
    fitted: Option<(Vec<f64>, f64, usize)>,
    marker: core::marker::PhantomData<F>,
}

impl<F: Float> LogisticRegression<F> {
    /// An unfitted model with inverse regularisation strength `c` (must be positive).
    /// Defaults: at most 100 Newton iterations, gradient tolerance `1e-4`.
    pub fn new(c: f64) -> Self {
        Self {
            c,
            max_iter: 100,
            tol: 1e-4,
            fitted: None,
            marker: core::marker::PhantomData,
        }
    }

    /// Sets the iteration limit (default 100).
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the largest absolute gradient component that counts as converged (default `1e-4`).
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Fitted coefficients, or `None` before `fit`.
    pub fn coef(&self) -> Option<&[f64]> {
        self.fitted.as_ref().map(|(w, _, _)| w.as_slice())
    }

    /// Fitted intercept, or `None` before `fit`.
    pub fn intercept(&self) -> Option<f64> {
        self.fitted.as_ref().map(|(_, b, _)| *b)
    }

    /// Newton iterations used, or `None` before `fit`.
    pub fn n_iter(&self) -> Option<usize> {
        self.fitted.as_ref().map(|(_, _, it)| *it)
    }

    /// Probability of class 1 for each row.
    pub fn predict_proba<X: ArrayView<Elem = F>>(&self, x: X) -> Result<Vec<F>, Error> {
        let (w, b, _) = self.fitted.as_ref().ok_or(Error::NotFitted)?;
        if x.cols() != w.len() {
            return Err(Error::Shape(ShapeError {
                rows: x.rows(),
                cols: x.cols(),
                reason: "feature count differs from the fitted model",
            }));
        }
        if !x.all_finite() {
            return Err(Error::NonFinite { what: "X" });
        }
        Ok((0..x.rows())
            .map(|r| {
                let z: f64 = (0..x.cols())
                    .map(|c| x.get(r, c).to_f64() * w[c])
                    .sum::<f64>()
                    + b;
                F::from_f64(sigmoid(z))
            })
            .collect())
    }
}

impl<F: Float> SupervisedEstimator for LogisticRegression<F> {
    type Float = F;
    type FitError = Error;

    fn fit<X, Y>(&mut self, x: X, y: Y) -> Result<(), Error>
    where
        X: ArrayView<Elem = F>,
        Y: TargetView<Elem = F>,
    {
        self.fitted = None;
        if !(self.c.is_finite() && self.c > 0.0) {
            return Err(Error::InvalidParameter {
                name: "c",
                reason: "must be finite and positive",
            });
        }
        let (rows, cols) = (x.rows(), x.cols());
        if rows == 0 || cols == 0 {
            return Err(Error::EmptyInput);
        }
        requires!(rows >= 1, "logistic.rows_positive");
        requires!(self.c > 0.0, "logistic.c_positive");
        if y.len() != rows {
            return Err(Error::Shape(ShapeError {
                rows,
                cols,
                reason: "target length differs from the number of rows",
            }));
        }
        if !x.all_finite() {
            return Err(Error::NonFinite { what: "X" });
        }
        let labels: Vec<f64> = (0..rows).map(|i| y.get(i).to_f64()).collect();
        if labels.iter().any(|&v| v != 0.0 && v != 1.0) {
            return Err(Error::InvalidParameter {
                name: "y",
                reason: "labels must be 0 or 1",
            });
        }
        if labels.iter().all(|&v| v == labels[0]) {
            return Err(Error::InvalidParameter {
                name: "y",
                reason: "both classes must be present",
            });
        }

        // Parameters θ = (w_0 … w_{d-1}, b); the design is [X | 1].
        let p = cols + 1;
        let mut design = Vec::with_capacity(rows * p);
        for r in 0..rows {
            for c in 0..cols {
                design.push(x.get(r, c).to_f64());
            }
            design.push(1.0);
        }
        let penalty =
            |theta: &[f64]| -> f64 { 0.5 * theta[..cols].iter().map(|w| w * w).sum::<f64>() };
        let objective = |theta: &[f64]| -> f64 {
            let loss: f64 = (0..rows)
                .map(|i| {
                    let row = &design[i * p..(i + 1) * p];
                    let z: f64 = row.iter().zip(theta).map(|(a, t)| a * t).sum();
                    softplus(z) - labels[i] * z
                })
                .sum();
            penalty(theta) + self.c * loss
        };

        let mut theta = vec![0.0_f64; p];
        let mut converged_at = None;
        for iter in 0..=self.max_iter {
            // Gradient and Hessian of the objective at θ.
            let mut grad = vec![0.0_f64; p];
            let mut hess = vec![0.0_f64; p * p];
            for i in 0..rows {
                let row = &design[i * p..(i + 1) * p];
                let z: f64 = row.iter().zip(&theta).map(|(a, t)| a * t).sum();
                let s = sigmoid(z);
                let weight = s * (1.0 - s);
                for a in 0..p {
                    grad[a] += self.c * row[a] * (s - labels[i]);
                    for b in 0..p {
                        hess[a * p + b] += self.c * weight * row[a] * row[b];
                    }
                }
            }
            for j in 0..cols {
                grad[j] += theta[j];
                hess[j * p + j] += 1.0;
            }
            let gradient_norm = grad.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
            if gradient_norm <= self.tol {
                // The stationarity condition holds at the returned point.
                ensures!(gradient_norm <= self.tol, "logistic.stationary");
                converged_at = Some(iter);
                break;
            }
            if iter == self.max_iter {
                break;
            }

            // Newton direction, with a tiny ridge if the Hessian is numerically singular.
            let rhs: Vec<f64> = grad.iter().map(|g| -g).collect();
            let step = cholesky_solve(&hess, &rhs, p)
                .or_else(|| {
                    let mut h = hess.clone();
                    for a in 0..p {
                        h[a * p + a] += 1e-10;
                    }
                    cholesky_solve(&h, &rhs, p)
                })
                .ok_or(Error::InvalidParameter {
                    name: "c",
                    reason: "the Newton system is singular; the data may be separable",
                })?;

            // Backtracking line search on the objective (Armijo condition).
            let f0 = objective(&theta);
            let slope: f64 = grad.iter().zip(&step).map(|(g, s)| g * s).sum();
            let mut t = 1.0_f64;
            let mut candidate = theta.clone();
            loop {
                for a in 0..p {
                    candidate[a] = theta[a] + t * step[a];
                }
                if objective(&candidate) <= f0 + 1e-4 * t * slope || t < 1e-12 {
                    break;
                }
                t *= 0.5;
            }
            theta = candidate;
        }

        let converged_at = converged_at.ok_or(Error::InvalidParameter {
            name: "max_iter",
            reason: "did not converge within the iteration limit",
        })?;
        let intercept = theta[cols];
        theta.truncate(cols);
        self.fitted = Some((theta, intercept, converged_at));
        Ok(())
    }
}

impl<F: Float> Predictor for LogisticRegression<F> {
    type Float = F;
    type Output = Vec<F>;

    /// Predicted class labels, `0` or `1`: class 1 when its probability is at least 1/2.
    fn predict<X>(&self, x: X) -> Result<Vec<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        Ok(self
            .predict_proba(x)?
            .into_iter()
            .map(|p| {
                if p.to_f64() >= 0.5 {
                    F::from_f64(1.0)
                } else {
                    F::from_f64(0.0)
                }
            })
            .collect())
    }
}
