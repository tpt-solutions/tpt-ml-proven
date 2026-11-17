// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Truncated singular value decomposition.
//!
//! Unlike [`crate::Pca`], the data is not centred, so this works on sparse-friendly data
//! and does not need the means. The right singular vectors are the eigenvectors of
//! `XᵀX`, and the singular values are the square roots of its eigenvalues. This is exact
//! up to rounding, and it matches scikit-learn's `algorithm="arpack"` solver.
//!
//! Sign convention: as in [`crate::Pca`], the largest-magnitude entry of each component
//! is positive.

use tpt_ml_proven_core::error::Error;
use tpt_ml_proven_core::{ArrayView, Dense, Float, Transformer, UnsupervisedEstimator};

use crate::eigen::symmetric_eigen;

/// Truncated SVD keeping the top `n_components` singular vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct TruncatedSvd<F: Float> {
    n_components: usize,
    fitted: Option<Fitted>,
    marker: core::marker::PhantomData<F>,
}

#[derive(Debug, Clone, PartialEq)]
struct Fitted {
    components: Vec<Vec<f64>>,
    singular_values: Vec<f64>,
    cols: usize,
}

impl<F: Float> TruncatedSvd<F> {
    /// An unfitted model keeping `n_components` components.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            fitted: None,
            marker: core::marker::PhantomData,
        }
    }

    /// Right singular vectors, one per row (`n_components x n_features`).
    pub fn components(&self) -> Option<Vec<Vec<f64>>> {
        self.fitted.as_ref().map(|f| f.components.clone())
    }

    /// Singular values in decreasing order.
    pub fn singular_values(&self) -> Option<&[f64]> {
        self.fitted.as_ref().map(|f| f.singular_values.as_slice())
    }
}

impl<F: Float> UnsupervisedEstimator for TruncatedSvd<F> {
    type Float = F;
    type FitError = Error;

    fn fit<X>(&mut self, x: X) -> Result<(), Error>
    where
        X: ArrayView<Elem = F>,
    {
        self.fitted = None;
        let (n, d) = (x.rows(), x.cols());
        if n == 0 || d == 0 {
            return Err(Error::EmptyInput);
        }
        if self.n_components == 0 || self.n_components > d {
            return Err(Error::InvalidParameter {
                name: "n_components",
                reason: "must be between 1 and the number of features",
            });
        }
        if !x.all_finite() {
            return Err(Error::NonFinite { what: "X" });
        }
        let mut gram = vec![0.0_f64; d * d];
        for r in 0..n {
            for i in 0..d {
                let xi = x.get(r, i).to_f64();
                for j in i..d {
                    gram[i * d + j] += xi * x.get(r, j).to_f64();
                }
            }
        }
        for i in 0..d {
            for j in 0..i {
                gram[i * d + j] = gram[j * d + i];
            }
        }
        let eig = symmetric_eigen(&gram, d);
        let k = self.n_components;
        let mut components = Vec::with_capacity(k);
        let mut singular_values = Vec::with_capacity(k);
        for i in 0..k {
            let mut v = eig.vectors[i].clone();
            let mut best = 0;
            for (j, x) in v.iter().enumerate() {
                if x.abs() > v[best].abs() {
                    best = j;
                }
            }
            if v[best] < 0.0 {
                v.iter_mut().for_each(|x| *x = -*x);
            }
            components.push(v);
            singular_values.push(eig.values[i].max(0.0).sqrt());
        }
        self.fitted = Some(Fitted {
            components,
            singular_values,
            cols: d,
        });
        Ok(())
    }
}

impl<F: Float> Transformer for TruncatedSvd<F> {
    type Float = F;
    type Output = Dense<F>;

    /// Projection onto the components, `X·Vₖ` (`rows x n_components`).
    fn transform<X>(&self, x: X) -> Result<Dense<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        let fitted = self.fitted.as_ref().ok_or(Error::NotFitted)?;
        if x.cols() != fitted.cols {
            return Err(Error::Shape(tpt_ml_proven_core::ShapeError {
                rows: x.rows(),
                cols: x.cols(),
                reason: "feature count differs from the fitted model",
            }));
        }
        let k = fitted.components.len();
        let mut out = Vec::with_capacity(x.rows() * k);
        for r in 0..x.rows() {
            for comp in &fitted.components {
                let s: f64 = (0..x.cols()).map(|c| x.get(r, c).to_f64() * comp[c]).sum();
                out.push(F::from_f64(s));
            }
        }
        Ok(Dense::from_vec(out, x.rows(), k)?)
    }

    fn fit_transform<X>(&mut self, x: X) -> Result<Dense<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        UnsupervisedEstimator::fit(self, &x)?;
        self.transform(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ml_proven_core::DenseView;

    #[test]
    fn rank_one_matrix_has_one_singular_value() {
        // Rows are multiples of (1, 2): singular value sqrt(1+4)·sqrt(1+4+9)·… with
        // X = u·vᵀ, so the single singular value is ‖u‖·‖v‖.
        let data = [1.0_f64, 2.0, 2.0, 4.0, 3.0, 6.0];
        let mut svd = TruncatedSvd::<f64>::new(1);
        let out = svd
            .fit_transform(DenseView::new(&data, 3, 2).unwrap())
            .unwrap();
        let sv = svd.singular_values().unwrap()[0];
        // ‖X‖_F² = 1+4+4+16+9+36 = 70 = sv².
        assert!((sv * sv - 70.0).abs() < 1e-9);
        // Projection norms equal the singular value (one component captures everything).
        let norm: f64 = out.into_vec().iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - sv).abs() < 1e-9);
    }

    #[test]
    fn invalid_component_count_is_rejected() {
        let data = [1.0_f64, 2.0, 3.0, 4.0];
        let mut svd = TruncatedSvd::<f64>::new(3);
        assert!(matches!(
            UnsupervisedEstimator::fit(&mut svd, DenseView::new(&data, 2, 2).unwrap()),
            Err(Error::InvalidParameter {
                name: "n_components",
                ..
            })
        ));
    }
}
