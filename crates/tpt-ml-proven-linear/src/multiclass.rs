// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Multinomial (softmax) logistic regression for two or more classes, fitted by damped
//! Newton iterations.
//!
//! The objective is the one scikit-learn minimises with `C` for `multi_class="multinomial"`:
//!
//! `½ Σ_k ‖w_k‖² + C · Σᵢ [logsumexp(zᵢ) − z_{i,yᵢ}]`, with `z_{i,k} = w_kᵀxᵢ + b_k`.
//!
//! The intercepts `b_k` are not penalised, so the objective is invariant to adding one
//! constant to every `b_k`. Compare fitted probabilities and coefficients, not intercepts.
//! Class labels are arbitrary numbers; each distinct value is a class.

use tpt_ml_proven_core::error::{Error, ShapeError};
use tpt_ml_proven_core::{
    ensures, requires, ArrayView, Float, Predictor, SupervisedEstimator, TargetView,
};

use crate::cholesky_solve;

/// Softmax of `z`, computed stably, in place.
fn softmax(z: &mut [f64]) {
    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut total = 0.0;
    for v in z.iter_mut() {
        *v = (*v - m).exp();
        total += *v;
    }
    for v in z.iter_mut() {
        *v /= total;
    }
}

/// Multinomial logistic regression with an L2 penalty controlled by `C`.
#[derive(Debug, Clone, PartialEq)]
pub struct MulticlassLogistic<F: Float> {
    c: f64,
    max_iter: usize,
    tol: f64,
    /// Classes, sorted; `coef` holds `n_classes × n_features`, row-major.
    fitted: Option<Fitted>,
    marker: core::marker::PhantomData<F>,
}

#[derive(Debug, Clone, PartialEq)]
struct Fitted {
    classes: Vec<f64>,
    coef: Vec<f64>,
    intercept: Vec<f64>,
    cols: usize,
    n_iter: usize,
}

impl<F: Float> MulticlassLogistic<F> {
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

    /// The classes seen in training, in ascending order, or `None` before `fit`.
    pub fn classes(&self) -> Option<&[f64]> {
        self.fitted.as_ref().map(|f| f.classes.as_slice())
    }

    /// Fitted weights, `n_classes × n_features` row-major, or `None` before `fit`.
    pub fn coef(&self) -> Option<&[f64]> {
        self.fitted.as_ref().map(|f| f.coef.as_slice())
    }

    /// Fitted intercepts, one per class, or `None` before `fit`.
    pub fn intercept(&self) -> Option<&[f64]> {
        self.fitted.as_ref().map(|f| f.intercept.as_slice())
    }

    /// Newton iterations used, or `None` before `fit`.
    pub fn n_iter(&self) -> Option<usize> {
        self.fitted.as_ref().map(|f| f.n_iter)
    }

    /// Class probabilities, `rows × n_classes` row-major, in the order of [`Self::classes`].
    pub fn predict_proba<X: ArrayView<Elem = F>>(&self, x: X) -> Result<Vec<F>, Error> {
        let f = self.fitted.as_ref().ok_or(Error::NotFitted)?;
        let k = f.classes.len();
        if x.cols() != f.cols {
            return Err(Error::Shape(ShapeError {
                rows: x.rows(),
                cols: x.cols(),
                reason: "feature count differs from the fitted model",
            }));
        }
        if !x.all_finite() {
            return Err(Error::NonFinite { what: "X" });
        }
        let mut out = Vec::with_capacity(x.rows() * k);
        let mut z = vec![0.0_f64; k];
        for r in 0..x.rows() {
            for c in 0..k {
                z[c] = f.intercept[c]
                    + (0..f.cols)
                        .map(|j| x.get(r, j).to_f64() * f.coef[c * f.cols + j])
                        .sum::<f64>();
            }
            softmax(&mut z);
            out.extend(z.iter().map(|&p| F::from_f64(p)));
        }
        Ok(out)
    }
}

impl<F: Float> SupervisedEstimator for MulticlassLogistic<F> {
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
        requires!(rows >= 1, "multiclass_logistic.rows_positive");
        requires!(self.c > 0.0, "multiclass_logistic.c_positive");

        let labels: Vec<f64> = (0..rows).map(|i| y.get(i).to_f64()).collect();
        if labels.iter().any(|v| !v.is_finite()) {
            return Err(Error::NonFinite { what: "y" });
        }
        let mut classes = labels.clone();
        classes.sort_by(f64::total_cmp);
        classes.dedup();
        if classes.len() < 2 {
            return Err(Error::InvalidParameter {
                name: "y",
                reason: "at least two classes are required",
            });
        }
        let k = classes.len();
        let class_of: Vec<usize> = labels
            .iter()
            .map(|v| {
                classes
                    .binary_search_by(|c| c.total_cmp(v))
                    .expect("label is a class")
            })
            .collect();

        // Parameters: for class c, block [c·(d+1), (c+1)·(d+1)) holds w_c (d values) then b_c.
        let q = cols + 1;
        let p = k * q;
        let mut design = Vec::with_capacity(rows * q);
        for r in 0..rows {
            for c in 0..cols {
                design.push(x.get(r, c).to_f64());
            }
            design.push(1.0);
        }
        let objective = |theta: &[f64]| -> f64 {
            let mut loss = 0.0;
            let mut z = vec![0.0_f64; k];
            for i in 0..rows {
                let row = &design[i * q..(i + 1) * q];
                for c in 0..k {
                    z[c] = row
                        .iter()
                        .zip(&theta[c * q..(c + 1) * q])
                        .map(|(a, t)| a * t)
                        .sum();
                }
                let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let lse = m + z.iter().map(|v| (v - m).exp()).sum::<f64>().ln();
                loss += lse - z[class_of[i]];
            }
            let penalty: f64 = (0..k)
                .map(|c| {
                    theta[c * q..c * q + cols]
                        .iter()
                        .map(|w| w * w)
                        .sum::<f64>()
                })
                .sum();
            0.5 * penalty + self.c * loss
        };

        // Gauge: adding one constant to every intercept leaves the objective unchanged, so the
        // last class's intercept is pinned at 0. This makes the Hessian positive definite.
        let fixed = (k - 1) * q + cols;
        let mut theta = vec![0.0_f64; p];
        let mut n_iter = None;
        for iter in 0..=self.max_iter {
            // Gradient and Hessian of the objective at theta.
            let mut grad = vec![0.0_f64; p];
            let mut hess = vec![0.0_f64; p * p];
            let mut z = vec![0.0_f64; k];
            for i in 0..rows {
                let row = &design[i * q..(i + 1) * q];
                for c in 0..k {
                    z[c] = row
                        .iter()
                        .zip(&theta[c * q..(c + 1) * q])
                        .map(|(a, t)| a * t)
                        .sum();
                }
                softmax(&mut z);
                for c in 0..k {
                    let target = if class_of[i] == c { 1.0 } else { 0.0 };
                    let resid = self.c * (z[c] - target);
                    for a in 0..q {
                        grad[c * q + a] += resid * row[a];
                    }
                    for d in 0..k {
                        // Block (c, d) of the Hessian: C · p_c (δ_cd − p_d) x xᵀ.
                        let w = self.c * z[c] * ((c == d) as u8 as f64 - z[d]);
                        for a in 0..q {
                            for b in 0..q {
                                hess[(c * q + a) * p + d * q + b] += w * row[a] * row[b];
                            }
                        }
                    }
                }
            }
            for c in 0..k {
                for a in 0..cols {
                    grad[c * q + a] += theta[c * q + a];
                    hess[(c * q + a) * p + c * q + a] += 1.0;
                }
            }
            // The last class's intercept is fixed at 0 (see the module docs), so its gradient
            // component is not a condition to meet.
            grad[fixed] = 0.0;
            let gradient_norm = grad.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
            if gradient_norm <= self.tol {
                ensures!(gradient_norm <= self.tol, "multiclass_logistic.stationary");
                n_iter = Some(iter);
                break;
            }
            if iter == self.max_iter {
                break;
            }

            // The intercept directions are flat (see the module docs), so the Hessian can be
            // singular; a tiny ridge keeps the Cholesky factorisation defined there.
            for j in 0..p {
                hess[fixed * p + j] = 0.0;
                hess[j * p + fixed] = 0.0;
            }
            hess[fixed * p + fixed] = 1.0;
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
                    reason: "the Newton system is singular",
                })?;

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
            theta[fixed] = 0.0;
        }

        let n_iter = n_iter.ok_or(Error::InvalidParameter {
            name: "max_iter",
            reason: "did not converge within the iteration limit",
        })?;
        let mut coef = Vec::with_capacity(k * cols);
        let mut intercept = Vec::with_capacity(k);
        for c in 0..k {
            coef.extend_from_slice(&theta[c * q..c * q + cols]);
            intercept.push(theta[c * q + cols]);
        }
        self.fitted = Some(Fitted {
            classes,
            coef,
            intercept,
            cols,
            n_iter,
        });
        Ok(())
    }
}

impl<F: Float> Predictor for MulticlassLogistic<F> {
    type Float = F;
    type Output = Vec<F>;

    /// The most probable class for each row. Ties go to the smaller class.
    fn predict<X>(&self, x: X) -> Result<Vec<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        let f = self.fitted.as_ref().ok_or(Error::NotFitted)?;
        let k = f.classes.len();
        let proba = self.predict_proba(x)?;
        Ok(proba
            .chunks(k)
            .map(|row| {
                let mut best = 0;
                for c in 1..k {
                    if row[c].to_f64() > row[best].to_f64() {
                        best = c;
                    }
                }
                F::from_f64(f.classes[best])
            })
            .collect())
    }
}
