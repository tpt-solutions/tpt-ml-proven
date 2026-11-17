// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Principal component analysis.
//!
//! The contract for this algorithm is `docs/contracts/pca.tptc`. Each clause has a
//! runtime check named `pca.<id>` in this file.
//!
//! Computation is done in `f64` whatever the element type. The covariance matrix
//! is decomposed with a Jacobi solver, and the results are rounded to `F` at the
//! end. The contract checks run on the `f64` values, before rounding.
//!
//! Sign convention: in each component, the entry with the largest absolute value
//! is positive (the first such entry on ties). This makes the output deterministic.

use tpt_ml_proven_core::error::Error;
use tpt_ml_proven_core::{
    ensures, requires, ArrayView, Dense, Float, Transformer, UnsupervisedEstimator,
};

use crate::eigen::symmetric_eigen;

/// Tolerance for the orthogonality and unit-norm contract checks.
const ORTHO_TOL: f64 = 1e-8;

/// PCA estimator. Build with [`Pca::new`], then call `fit`.
#[derive(Debug, Clone, PartialEq)]
pub struct Pca<F: Float> {
    n_components: usize,
    fitted: Option<Fitted<F>>,
}

#[derive(Debug, Clone, PartialEq)]
struct Fitted<F: Float> {
    mean: Vec<F>,
    components: Dense<F>,
    explained_variance: Vec<F>,
    explained_variance_ratio: Vec<F>,
    total_variance: F,
}

impl<F: Float> Pca<F> {
    /// An unfitted PCA that keeps `n_components` components.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            fitted: None,
        }
    }

    /// Requested number of components.
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Principal axes, one per row (`n_components x n_features`), or `None` before `fit`.
    pub fn components(&self) -> Option<&Dense<F>> {
        self.fitted.as_ref().map(|f| &f.components)
    }

    /// Per-feature mean removed before projection, or `None` before `fit`.
    pub fn mean(&self) -> Option<&[F]> {
        self.fitted.as_ref().map(|f| f.mean.as_slice())
    }

    /// Variance along each kept component, in descending order.
    pub fn explained_variance(&self) -> Option<&[F]> {
        self.fitted
            .as_ref()
            .map(|f| f.explained_variance.as_slice())
    }

    /// Fraction of the total variance each kept component explains. All zero if the
    /// data has no variance at all.
    pub fn explained_variance_ratio(&self) -> Option<&[F]> {
        self.fitted
            .as_ref()
            .map(|f| f.explained_variance_ratio.as_slice())
    }

    /// Total variance of the data (the sum of all eigenvalues, kept or not).
    pub fn total_variance(&self) -> Option<F> {
        self.fitted.as_ref().map(|f| f.total_variance)
    }

    /// Projects `x` onto the kept components (`rows x n_components`).
    pub fn transform_view<X: ArrayView<Elem = F>>(&self, x: &X) -> Result<Dense<F>, Error> {
        let fitted = self.fitted.as_ref().ok_or(Error::NotFitted)?;
        let (n, d) = (x.rows(), x.cols());
        check_cols(n, d, fitted.mean.len())?;
        if !x.all_finite() {
            return Err(Error::NonFinite { what: "X" });
        }
        let k = fitted.components.rows();
        let mut out = Vec::with_capacity(n * k);
        for r in 0..n {
            for c in 0..k {
                let mut acc = F::ZERO;
                for j in 0..d {
                    let centred = x.get(r, j) - fitted.mean[j];
                    acc = acc + centred * fitted.components.get(c, j);
                }
                out.push(acc);
            }
        }
        Ok(Dense::from_vec(out, n, k)?)
    }

    /// Maps component scores back to feature space (`rows x n_features`). Exact when
    /// all components are kept; otherwise the discarded directions are lost.
    pub fn inverse_transform<X: ArrayView<Elem = F>>(&self, z: &X) -> Result<Dense<F>, Error> {
        let fitted = self.fitted.as_ref().ok_or(Error::NotFitted)?;
        let k = fitted.components.rows();
        let d = fitted.mean.len();
        check_cols(z.rows(), z.cols(), k)?;
        if !z.all_finite() {
            return Err(Error::NonFinite { what: "Z" });
        }
        let n = z.rows();
        let mut out = Vec::with_capacity(n * d);
        for r in 0..n {
            for j in 0..d {
                let mut acc = fitted.mean[j];
                for c in 0..k {
                    acc = acc + z.get(r, c) * fitted.components.get(c, j);
                }
                out.push(acc);
            }
        }
        Ok(Dense::from_vec(out, n, d)?)
    }
}

fn check_cols(rows: usize, cols: usize, expected: usize) -> Result<(), Error> {
    if cols != expected {
        return Err(Error::Shape(tpt_ml_proven_core::ShapeError {
            rows,
            cols,
            reason: "feature count differs from the fitted model",
        }));
    }
    Ok(())
}

impl<F: Float> UnsupervisedEstimator for Pca<F> {
    type Float = F;
    type FitError = Error;

    fn fit<X>(&mut self, x: X) -> Result<(), Error>
    where
        X: ArrayView<Elem = F>,
    {
        self.fitted = None;
        let (n, d) = (x.rows(), x.cols());
        let k = self.n_components;
        validate(n, d, k)?;
        if !x.all_finite() {
            return Err(Error::NonFinite { what: "X" });
        }
        requires!(n >= 2, "pca.rows_min");
        requires!(k >= 1, "pca.components_positive");
        requires!(k <= n, "pca.components_le_rows");
        requires!(k <= d, "pca.components_le_cols");

        // Work in f64 throughout.
        let data: Vec<f64> = (0..n)
            .flat_map(|r| (0..d).map(move |j| (r, j)))
            .map(|(r, j)| x.get(r, j).to_f64())
            .collect();
        let mean: Vec<f64> = (0..d)
            .map(|j| (0..n).map(|r| data[r * d + j]).sum::<f64>() / n as f64)
            .collect();
        let centred: Vec<f64> = data
            .iter()
            .enumerate()
            .map(|(idx, v)| v - mean[idx % d])
            .collect();

        // Covariance (unbiased, n - 1 denominator).
        let mut cov = vec![0.0_f64; d * d];
        for p in 0..d {
            for q in p..d {
                let s: f64 = (0..n)
                    .map(|r| centred[r * d + p] * centred[r * d + q])
                    .sum();
                let v = s / (n - 1) as f64;
                cov[p * d + q] = v;
                cov[q * d + p] = v;
            }
        }

        let eig = symmetric_eigen(&cov, d);
        // Rounding can leave tiny negative eigenvalues. Variance is never negative.
        let variances: Vec<f64> = eig.values.iter().map(|&v| v.max(0.0)).collect();
        let total: f64 = variances.iter().sum();

        let mut comps: Vec<Vec<f64>> = eig.vectors.into_iter().take(k).collect();
        for v in &mut comps {
            flip_sign(v);
        }
        let explained: Vec<f64> = variances[..k].to_vec();

        ensures!(k <= n, "pca.components_within_rows");
        ensures!(orthogonal(&comps), "pca.orthogonality");
        ensures!(unit_norms(&comps), "pca.unit_norm");
        ensures!(
            explained.iter().sum::<f64>() <= total + 1e-9 * (1.0 + total),
            "pca.variance_bound"
        );

        let flat: Vec<F> = comps
            .iter()
            .flat_map(|v| v.iter().map(|&x| F::from_f64(x)))
            .collect();
        self.fitted = Some(Fitted {
            mean: mean.iter().map(|&m| F::from_f64(m)).collect(),
            components: Dense::from_vec(flat, k, d)?,
            explained_variance: explained.iter().map(|&e| F::from_f64(e)).collect(),
            explained_variance_ratio: explained
                .iter()
                .map(|&e| F::from_f64(if total > 0.0 { e / total } else { 0.0 }))
                .collect(),
            total_variance: F::from_f64(total),
        });
        Ok(())
    }
}

impl<F: Float> Transformer for Pca<F> {
    type Float = F;
    type Output = Dense<F>;

    fn transform<X>(&self, x: X) -> Result<Dense<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        self.transform_view(&x)
    }

    fn fit_transform<X>(&mut self, x: X) -> Result<Dense<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        UnsupervisedEstimator::fit(self, &x)?;
        self.transform_view(&x)
    }
}

fn validate(n: usize, d: usize, k: usize) -> Result<(), Error> {
    if n < 2 {
        return Err(Error::InvalidParameter {
            name: "rows",
            reason: "PCA needs at least 2 samples",
        });
    }
    if d == 0 {
        return Err(Error::EmptyInput);
    }
    if k == 0 {
        return Err(Error::InvalidParameter {
            name: "n_components",
            reason: "must be at least 1",
        });
    }
    if k > n.min(d) {
        return Err(Error::InvalidParameter {
            name: "n_components",
            reason: "exceeds min(rows, cols)",
        });
    }
    Ok(())
}

/// Makes the largest-magnitude entry positive. The first such entry wins a tie.
fn flip_sign(v: &mut [f64]) {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if x.abs() > v[best].abs() {
            best = i;
        }
    }
    if v[best] < 0.0 {
        v.iter_mut().for_each(|x| *x = -*x);
    }
}

fn orthogonal(comps: &[Vec<f64>]) -> bool {
    for i in 0..comps.len() {
        for j in i + 1..comps.len() {
            let dot: f64 = comps[i].iter().zip(&comps[j]).map(|(a, b)| a * b).sum();
            if dot.abs() > ORTHO_TOL {
                return false;
            }
        }
    }
    true
}

fn unit_norms(comps: &[Vec<f64>]) -> bool {
    comps.iter().all(|v| {
        let norm: f64 = v.iter().map(|x| x * x).sum();
        (norm - 1.0).abs() <= ORTHO_TOL
    })
}
