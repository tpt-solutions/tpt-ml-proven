// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Linear regression models for tpt-ml-proven.
//!
//! * [`Ridge`] minimises `‖y − Xw − b‖² + α‖w‖²`, solved in closed form.
//! * [`ElasticNet`] minimises `(1/2n)‖y − Xw − b‖² + α·ρ‖w‖₁ + ½·α·(1 − ρ)‖w‖²` by cyclic
//!   coordinate descent. [`Lasso`] is the case `ρ = 1`.
//!
//! These are the objectives scikit-learn uses, so coefficients can be compared with it.
//! * [`LogisticRegression`] is binary classification with an L2 penalty, fitted by Newton
//!   iterations on the objective scikit-learn minimises (see the `logistic` module).
//!
//! Computation is in `f64`, and the intercept is fitted by centring the data.

mod logistic;
mod multiclass;
mod persist;

pub use logistic::LogisticRegression;
pub use multiclass::MulticlassLogistic;

use tpt_ml_proven_core::error::{Error, ShapeError};
use tpt_ml_proven_core::{
    ensures, requires, ArrayView, Float, Predictor, SupervisedEstimator, TargetView,
};

/// Centred design matrix and target, with the means needed to recover the intercept.
struct Centred {
    x: Vec<f64>,
    y: Vec<f64>,
    rows: usize,
    cols: usize,
    x_mean: Vec<f64>,
    y_mean: f64,
}

fn centre<F: Float, X: ArrayView<Elem = F>, Y: TargetView<Elem = F>>(
    x: &X,
    y: &Y,
) -> Result<Centred, Error> {
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
    let n = rows as f64;
    let mut raw = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            raw.push(x.get(r, c).to_f64());
        }
    }
    let raw_y: Vec<f64> = (0..rows).map(|i| y.get(i).to_f64()).collect();
    if raw_y.iter().any(|v| !v.is_finite()) {
        return Err(Error::NonFinite { what: "y" });
    }
    let x_mean: Vec<f64> = (0..cols)
        .map(|c| (0..rows).map(|r| raw[r * cols + c]).sum::<f64>() / n)
        .collect();
    let y_mean = raw_y.iter().sum::<f64>() / n;
    let x_c: Vec<f64> = raw
        .iter()
        .enumerate()
        .map(|(i, v)| v - x_mean[i % cols])
        .collect();
    let y_c: Vec<f64> = raw_y.iter().map(|v| v - y_mean).collect();
    Ok(Centred {
        x: x_c,
        y: y_c,
        rows,
        cols,
        x_mean,
        y_mean,
    })
}

/// Solves `a·x = b` for symmetric positive definite `a` (row-major, `d × d`) by Cholesky.
pub(crate) fn cholesky_solve(a: &[f64], b: &[f64], d: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0_f64; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut s = a[i * d + j];
            for k in 0..j {
                s -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i * d + i] = s.sqrt();
            } else {
                l[i * d + j] = s / l[j * d + j];
            }
        }
    }
    let mut z = vec![0.0_f64; d];
    for i in 0..d {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * d + k] * z[k];
        }
        z[i] = s / l[i * d + i];
    }
    let mut x = vec![0.0_f64; d];
    for i in (0..d).rev() {
        let mut s = z[i];
        for k in i + 1..d {
            s -= l[k * d + i] * x[k];
        }
        x[i] = s / l[i * d + i];
    }
    Some(x)
}

/// Largest KKT violation allowed at convergence in the elastic-net contract.
const KKT_SLACK: f64 = 1e-3;

/// Largest violation of the elastic-net optimality conditions for `coef`. For coefficient
/// `j` with gradient `g_j = −x_jᵀr/n + λ₂w_j`: `g_j + λ₁·sign(w_j) = 0` when `w_j ≠ 0`,
/// and `|g_j| ≤ λ₁` when `w_j = 0`.
fn kkt_violation(c: &Centred, coef: &[f64], lambda1: f64, lambda2: f64) -> f64 {
    let (n, d) = (c.rows, c.cols);
    let nf = n as f64;
    let resid: Vec<f64> = (0..n)
        .map(|r| c.y[r] - (0..d).map(|j| c.x[r * d + j] * coef[j]).sum::<f64>())
        .collect();
    (0..d)
        .map(|j| {
            let g =
                -(0..n).map(|r| c.x[r * d + j] * resid[r]).sum::<f64>() / nf + lambda2 * coef[j];
            if coef[j] != 0.0 {
                (g + lambda1 * coef[j].signum()).abs()
            } else {
                (g.abs() - lambda1).max(0.0)
            }
        })
        .fold(0.0_f64, f64::max)
}

fn soft_threshold(z: f64, t: f64) -> f64 {
    if z > t {
        z - t
    } else if z < -t {
        z + t
    } else {
        0.0
    }
}

/// Predictions `X·coef + intercept` for each row.
fn predict_rows<F: Float, X: ArrayView<Elem = F>>(
    x: &X,
    coef: &[f64],
    intercept: f64,
) -> Result<Vec<F>, Error> {
    if x.cols() != coef.len() {
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
            let s: f64 = (0..x.cols()).map(|c| x.get(r, c).to_f64() * coef[c]).sum();
            F::from_f64(s + intercept)
        })
        .collect())
}

// ---------------------------------------------------------------- Ridge

/// Ridge regression: least squares with an L2 penalty `α‖w‖²`.
#[derive(Debug, Clone, PartialEq)]
pub struct Ridge<F: Float> {
    alpha: f64,
    fitted: Option<(Vec<f64>, f64)>,
    marker: core::marker::PhantomData<F>,
}

impl<F: Float> Ridge<F> {
    /// An unfitted model with penalty `alpha` (must be non-negative).
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            fitted: None,
            marker: core::marker::PhantomData,
        }
    }

    /// Fitted coefficients, or `None` before `fit`.
    pub fn coef(&self) -> Option<&[f64]> {
        self.fitted.as_ref().map(|(c, _)| c.as_slice())
    }

    /// Fitted intercept, or `None` before `fit`.
    pub fn intercept(&self) -> Option<f64> {
        self.fitted.as_ref().map(|(_, b)| *b)
    }
}

impl<F: Float> SupervisedEstimator for Ridge<F> {
    type Float = F;
    type FitError = Error;

    fn fit<X, Y>(&mut self, x: X, y: Y) -> Result<(), Error>
    where
        X: ArrayView<Elem = F>,
        Y: TargetView<Elem = F>,
    {
        self.fitted = None;
        if !(self.alpha.is_finite() && self.alpha >= 0.0) {
            return Err(Error::InvalidParameter {
                name: "alpha",
                reason: "must be finite and non-negative",
            });
        }
        let c = centre(&x, &y)?;
        let d = c.cols;
        requires!(c.rows >= 1, "ridge.rows_positive");
        requires!(self.alpha >= 0.0, "ridge.alpha_nonneg");
        // (XᵀX + αI) w = Xᵀy, on centred data.
        let mut gram = vec![0.0_f64; d * d];
        let mut xty = vec![0.0_f64; d];
        for r in 0..c.rows {
            for i in 0..d {
                let xi = c.x[r * d + i];
                xty[i] += xi * c.y[r];
                for j in 0..d {
                    gram[i * d + j] += xi * c.x[r * d + j];
                }
            }
        }
        for i in 0..d {
            gram[i * d + i] += self.alpha;
        }
        let coef = cholesky_solve(&gram, &xty, d).ok_or(Error::InvalidParameter {
            name: "alpha",
            reason: "the penalised system is singular; use alpha > 0 with collinear data",
        })?;
        // The normal equations hold at the solution: (XᵀX + αI) w − Xᵀy = 0, up to rounding.
        let scale = 1.0 + xty.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let normal_residual = (0..d)
            .map(|i| {
                let lhs: f64 = (0..d).map(|j| gram[i * d + j] * coef[j]).sum();
                (lhs - xty[i]).abs()
            })
            .fold(0.0_f64, f64::max);
        ensures!(normal_residual <= 1e-8 * scale, "ridge.normal_equations");
        let intercept = c.y_mean - coef.iter().zip(&c.x_mean).map(|(w, m)| w * m).sum::<f64>();
        self.fitted = Some((coef, intercept));
        Ok(())
    }
}

impl<F: Float> Predictor for Ridge<F> {
    type Float = F;
    type Output = Vec<F>;

    fn predict<X>(&self, x: X) -> Result<Vec<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        let (coef, intercept) = self.fitted.as_ref().ok_or(Error::NotFitted)?;
        predict_rows(&x, coef, *intercept)
    }
}

// ---------------------------------------------------------------- ElasticNet and Lasso

/// Elastic net, fitted by cyclic coordinate descent.
#[derive(Debug, Clone, PartialEq)]
pub struct ElasticNet<F: Float> {
    alpha: f64,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
    fitted: Option<(Vec<f64>, f64, usize)>,
    marker: core::marker::PhantomData<F>,
}

impl<F: Float> ElasticNet<F> {
    /// An unfitted model. `l1_ratio` is ρ in `[0, 1]`: 1 is the lasso, 0 is ridge-like.
    pub fn new(alpha: f64, l1_ratio: f64) -> Self {
        Self {
            alpha,
            l1_ratio,
            max_iter: 1000,
            tol: 1e-4,
            fitted: None,
            marker: core::marker::PhantomData,
        }
    }

    /// Sets the iteration limit (default 1000).
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the largest coefficient change that counts as converged (default `1e-4`).
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Fitted coefficients, or `None` before `fit`.
    pub fn coef(&self) -> Option<&[f64]> {
        self.fitted.as_ref().map(|(c, _, _)| c.as_slice())
    }

    /// Fitted intercept, or `None` before `fit`.
    pub fn intercept(&self) -> Option<f64> {
        self.fitted.as_ref().map(|(_, b, _)| *b)
    }

    /// Coordinate-descent sweeps used, or `None` before `fit`.
    pub fn n_iter(&self) -> Option<usize> {
        self.fitted.as_ref().map(|(_, _, it)| *it)
    }
}

/// Lasso regression, a thin wrapper over [`ElasticNet`] with `l1_ratio = 1`.
#[derive(Debug, Clone, PartialEq)]
pub struct Lasso<F: Float>(ElasticNet<F>);

impl<F: Float> Lasso<F> {
    /// An unfitted lasso with penalty `alpha`.
    pub fn new(alpha: f64) -> Self {
        Self(ElasticNet::new(alpha, 1.0))
    }

    /// Sets the iteration limit (default 1000).
    pub fn with_max_iter(self, max_iter: usize) -> Self {
        Self(self.0.with_max_iter(max_iter))
    }

    /// Sets the convergence tolerance (default `1e-4`).
    pub fn with_tol(self, tol: f64) -> Self {
        Self(self.0.with_tol(tol))
    }

    /// Fitted coefficients, or `None` before `fit`.
    pub fn coef(&self) -> Option<&[f64]> {
        self.0.coef()
    }

    /// Fitted intercept, or `None` before `fit`.
    pub fn intercept(&self) -> Option<f64> {
        self.0.intercept()
    }
}

impl<F: Float> SupervisedEstimator for ElasticNet<F> {
    type Float = F;
    type FitError = Error;

    fn fit<X, Y>(&mut self, x: X, y: Y) -> Result<(), Error>
    where
        X: ArrayView<Elem = F>,
        Y: TargetView<Elem = F>,
    {
        self.fitted = None;
        if !(self.alpha.is_finite() && self.alpha >= 0.0) {
            return Err(Error::InvalidParameter {
                name: "alpha",
                reason: "must be finite and non-negative",
            });
        }
        if !(0.0..=1.0).contains(&self.l1_ratio) {
            return Err(Error::InvalidParameter {
                name: "l1_ratio",
                reason: "must be in [0, 1]",
            });
        }
        let c = centre(&x, &y)?;
        let (n, d) = (c.rows, c.cols);
        requires!(n >= 1, "elasticnet.rows_positive");
        requires!(self.alpha >= 0.0, "elasticnet.alpha_nonneg");
        requires!(self.l1_ratio >= 0.0, "elasticnet.l1_ratio_nonneg");
        requires!(self.l1_ratio <= 1.0, "elasticnet.l1_ratio_le_one");
        let nf = n as f64;
        let lambda1 = self.alpha * self.l1_ratio;
        let lambda2 = self.alpha * (1.0 - self.l1_ratio);
        let col_norm2: Vec<f64> = (0..d)
            .map(|j| (0..n).map(|r| c.x[r * d + j] * c.x[r * d + j]).sum::<f64>() / nf)
            .collect();

        let mut coef = vec![0.0_f64; d];
        // Residual y − X·coef, kept up to date as coefficients change.
        let mut resid = c.y.clone();
        let mut iterations = 0;
        let mut converged = false;
        for sweep in 0..self.max_iter {
            iterations = sweep + 1;
            let mut max_change = 0.0_f64;
            let mut max_coef = 0.0_f64;
            for j in 0..d {
                let old = coef[j];
                // Partial residual with coefficient j removed.
                let rho: f64 = (0..n)
                    .map(|r| c.x[r * d + j] * (resid[r] + c.x[r * d + j] * old))
                    .sum::<f64>()
                    / nf;
                let denom = col_norm2[j] + lambda2;
                let new = if denom == 0.0 {
                    0.0
                } else {
                    soft_threshold(rho, lambda1) / denom
                };
                if new != old {
                    for r in 0..n {
                        resid[r] -= c.x[r * d + j] * (new - old);
                    }
                    coef[j] = new;
                }
                max_change = max_change.max((new - old).abs());
                max_coef = max_coef.max(new.abs());
            }
            if max_change <= self.tol * max_coef {
                converged = true;
                break;
            }
        }
        ensures!(self.l1_ratio <= 1.0, "elasticnet.l1_ratio_within_unit");
        if converged {
            // At convergence the KKT conditions hold up to the stopping rule's slack.
            ensures!(
                kkt_violation(&c, &coef, lambda1, lambda2) <= KKT_SLACK,
                "elasticnet.kkt"
            );
        }
        let intercept = c.y_mean - coef.iter().zip(&c.x_mean).map(|(w, m)| w * m).sum::<f64>();
        self.fitted = Some((coef, intercept, iterations));
        Ok(())
    }
}

impl<F: Float> Predictor for ElasticNet<F> {
    type Float = F;
    type Output = Vec<F>;

    fn predict<X>(&self, x: X) -> Result<Vec<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        let (coef, intercept, _) = self.fitted.as_ref().ok_or(Error::NotFitted)?;
        predict_rows(&x, coef, *intercept)
    }
}

impl<F: Float> SupervisedEstimator for Lasso<F> {
    type Float = F;
    type FitError = Error;

    fn fit<X, Y>(&mut self, x: X, y: Y) -> Result<(), Error>
    where
        X: ArrayView<Elem = F>,
        Y: TargetView<Elem = F>,
    {
        SupervisedEstimator::fit(&mut self.0, x, y)
    }
}

impl<F: Float> Predictor for Lasso<F> {
    type Float = F;
    type Output = Vec<F>;

    fn predict<X>(&self, x: X) -> Result<Vec<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        Predictor::predict(&self.0, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ml_proven_core::DenseView;

    /// y = 2·x0 − 1·x1 + 3, exactly.
    fn exact_data() -> (Vec<f64>, Vec<f64>) {
        let x = vec![
            0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 1.0,
        ];
        let y: Vec<f64> = x.chunks(2).map(|r| 2.0 * r[0] - r[1] + 3.0).collect();
        (x, y)
    }

    #[test]
    fn ridge_with_tiny_alpha_recovers_exact_linear_relation() {
        let (x, y) = exact_data();
        let mut m = Ridge::<f64>::new(1e-10);
        SupervisedEstimator::fit(&mut m, DenseView::new(&x, 6, 2).unwrap(), &y[..]).unwrap();
        assert!((m.coef().unwrap()[0] - 2.0).abs() < 1e-6);
        assert!((m.coef().unwrap()[1] + 1.0).abs() < 1e-6);
        assert!((m.intercept().unwrap() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn lasso_zeroes_an_irrelevant_feature_at_a_strong_penalty() {
        let (x, y) = exact_data();
        let mut m = Lasso::<f64>::new(5.0).with_tol(1e-12).with_max_iter(10_000);
        SupervisedEstimator::fit(&mut m, DenseView::new(&x, 6, 2).unwrap(), &y[..]).unwrap();
        assert!(
            m.coef().unwrap().iter().all(|w| w.abs() < 1e-9),
            "{:?}",
            m.coef()
        );
        // With no coefficients the intercept is the mean of y.
        let mean: f64 = y.iter().sum::<f64>() / 6.0;
        assert!((m.intercept().unwrap() - mean).abs() < 1e-9);
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        let (x, y) = exact_data();
        let mut m = Ridge::<f64>::new(-1.0);
        assert!(matches!(
            SupervisedEstimator::fit(&mut m, DenseView::new(&x, 6, 2).unwrap(), &y[..]),
            Err(Error::InvalidParameter { name: "alpha", .. })
        ));
        let mut m = ElasticNet::<f64>::new(0.1, 1.5);
        assert!(matches!(
            SupervisedEstimator::fit(&mut m, DenseView::new(&x, 6, 2).unwrap(), &y[..]),
            Err(Error::InvalidParameter {
                name: "l1_ratio",
                ..
            })
        ));
    }

    #[test]
    fn predict_before_fit_and_feature_mismatch_are_errors() {
        let (x, y) = exact_data();
        let m = Ridge::<f64>::new(0.1);
        assert_eq!(
            Predictor::predict(&m, DenseView::new(&x, 6, 2).unwrap()).unwrap_err(),
            Error::NotFitted
        );
        let mut m = Ridge::<f64>::new(0.1);
        SupervisedEstimator::fit(&mut m, DenseView::new(&x, 6, 2).unwrap(), &y[..]).unwrap();
        assert!(matches!(
            Predictor::predict(&m, DenseView::new(&x[..3], 3, 1).unwrap()),
            Err(Error::Shape(_))
        ));
    }
}
