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

use tpt_ml_proven_core::error::{Error, ShapeError};
use tpt_ml_proven_core::{
    ensures, requires, ArrayView, Dense, Float, Transformer, UnsupervisedEstimator,
};

use crate::eigen::symmetric_eigen;

/// Truncated SVD keeping the top `n_components` singular vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct TruncatedSvd<F: Float> {
    n_components: usize,
    fitted: Option<Fitted<F>>,
    marker: core::marker::PhantomData<F>,
}

#[derive(Debug, Clone, PartialEq)]
struct Fitted<F: Float> {
    /// Right singular vectors, `n_components × n_features`.
    components: Dense<F>,
    singular_values: Vec<f64>,
    /// Population variance of each projected training column (divisor `n`), as in
    /// scikit-learn's `TruncatedSVD.explained_variance_`.
    explained_variance: Vec<F>,
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
    pub fn components(&self) -> Option<&Dense<F>> {
        self.fitted.as_ref().map(|f| &f.components)
    }

    /// Singular values in decreasing order.
    pub fn singular_values(&self) -> Option<&[f64]> {
        self.fitted.as_ref().map(|f| f.singular_values.as_slice())
    }

    /// Variance of each projected training column, `n_components` values.
    pub fn explained_variance(&self) -> Option<&[F]> {
        self.fitted
            .as_ref()
            .map(|f| f.explained_variance.as_slice())
    }

    /// Maps projections `Z` (`rows x n_components`) back to feature space, `Z·Vₖᵀ`.
    /// This is exact when `n_components` equals the rank of the data, and otherwise the
    /// best rank-`n_components` approximation.
    pub fn inverse_transform<Y: ArrayView<Elem = F>>(&self, z: &Y) -> Result<Dense<F>, Error> {
        let fitted = self.fitted.as_ref().ok_or(Error::NotFitted)?;
        let k = fitted.components.rows();
        let d = fitted.cols;
        if z.cols() != k {
            return Err(Error::Shape(ShapeError {
                rows: z.rows(),
                cols: z.cols(),
                reason: "component count differs from the fitted model",
            }));
        }
        if !z.all_finite() {
            return Err(Error::NonFinite { what: "Z" });
        }
        let n = z.rows();
        let mut out = Vec::with_capacity(n * d);
        for r in 0..n {
            for j in 0..d {
                let mut acc = F::ZERO;
                for c in 0..k {
                    acc = acc + z.get(r, c) * fitted.components.get(c, j);
                }
                out.push(acc);
            }
        }
        Ok(Dense::from_vec(out, n, d)?)
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
        requires!(n >= 1, "svd.rows_positive");
        requires!(self.n_components >= 1, "svd.components_positive");
        requires!(self.n_components <= d, "svd.components_le_cols");
        // XᵀX over the non-zeros of each row. The entries come in column order, so every
        // cell of the Gram matrix receives the same products in the same order as a dense loop.
        let mut gram = vec![0.0_f64; d * d];
        let mut nonzeros: Vec<(usize, f64)> = Vec::new();
        for r in 0..n {
            nonzeros.clear();
            x.for_each_nonzero(r, |c, v| nonzeros.push((c, v.to_f64())));
            for (a, &(i, xi)) in nonzeros.iter().enumerate() {
                for &(j, xj) in &nonzeros[a..] {
                    gram[i * d + j] += xi * xj;
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
        // Unit, mutually orthogonal components, and singular values in descending order.
        ensures!(
            components
                .iter()
                .enumerate()
                .all(|(i, a)| components.iter().enumerate().all(|(j, b)| {
                    let dot: f64 = a.iter().zip(b).map(|(p, q)| p * q).sum();
                    let want = if i == j { 1.0 } else { 0.0 };
                    (dot - want).abs() <= 1e-8
                })),
            "svd.orthonormal_components"
        );
        ensures!(
            singular_values.iter().all(|&s| s >= 0.0)
                && singular_values.windows(2).all(|w| w[0] >= w[1]),
            "svd.singular_values_sorted"
        );
        ensures!(k <= d, "svd.components_within_cols");

        // Population variance of each projected column, from the training data.
        let mut explained_variance = Vec::with_capacity(k);
        for comp in &components {
            let proj: Vec<f64> = (0..n)
                .map(|r| {
                    let mut s = 0.0_f64;
                    x.for_each_nonzero(r, |c, v| s += v.to_f64() * comp[c]);
                    s
                })
                .collect();
            let mean = proj.iter().sum::<f64>() / n as f64;
            let var = proj.iter().map(|p| (p - mean) * (p - mean)).sum::<f64>() / n as f64;
            explained_variance.push(F::from_f64(var));
        }
        let flat: Vec<F> = components
            .iter()
            .flat_map(|c| c.iter().map(|v| F::from_f64(*v)))
            .collect();
        self.fitted = Some(Fitted {
            components: Dense::from_vec(flat, k, d)?,
            singular_values,
            explained_variance,
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
        let k = fitted.components.rows();
        let mut out = Vec::with_capacity(x.rows() * k);
        for r in 0..x.rows() {
            let mut sums = vec![0.0_f64; k];
            x.for_each_nonzero(r, |j, v| {
                let v = v.to_f64();
                for (c, s) in sums.iter_mut().enumerate() {
                    *s += v * fitted.components.get(c, j).to_f64();
                }
            });
            out.extend(sums.into_iter().map(F::from_f64));
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

// ---------------------------------------------------------------- model files

use tpt_ml_proven_core::persist::{kind, Reader, Writer};

fn invalid(reason: &'static str) -> Error {
    Error::Format { reason }
}

impl<F: Float> TruncatedSvd<F> {
    /// Serialises the model (settings and fitted state) to a model file.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut w = Writer::for_float::<F>(kind::TRUNCATED_SVD);
        w.usize(self.n_components);
        match &self.fitted {
            None => w.bool(false),
            Some(f) => {
                w.bool(true);
                w.usize(f.cols);
                let rows: Vec<Vec<f64>> = (0..f.components.rows())
                    .map(|r| {
                        (0..f.cols)
                            .map(|c| f.components.get(r, c).to_f64())
                            .collect()
                    })
                    .collect();
                w.f64_rows(&rows);
                w.f64s(&f.singular_values);
                let variance: Vec<f64> = f.explained_variance.iter().map(|v| v.to_f64()).collect();
                w.f64s(&variance);
            }
        }
        Ok(w.finish())
    }

    /// Loads a model written by [`TruncatedSvd::to_bytes`], checking its shapes and that the
    /// components are unit length and mutually orthogonal.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(bytes, kind::TRUNCATED_SVD)?;
        let n_components = r.usize()?;
        let fitted = if r.bool()? {
            let cols = r.usize()?;
            let rows = r.f64_rows()?;
            let singular_values = r.f64s()?;
            let explained_variance = r.f64s()?;
            if n_components == 0 || rows.len() != n_components || n_components > cols {
                return Err(invalid("SVD component count does not match the file"));
            }
            if rows.iter().any(|row| row.len() != cols) {
                return Err(invalid(
                    "SVD component length differs from the feature count",
                ));
            }
            if singular_values.len() != n_components || explained_variance.len() != n_components {
                return Err(invalid("SVD value count differs from n_components"));
            }
            if singular_values.iter().any(|&s| s < 0.0)
                || singular_values.windows(2).any(|w| w[0] < w[1])
            {
                return Err(invalid("SVD singular values are negative or not sorted"));
            }
            for (i, a) in rows.iter().enumerate() {
                for (j, b) in rows.iter().enumerate().skip(i) {
                    let dot: f64 = a.iter().zip(b).map(|(p, q)| p * q).sum();
                    let want = if i == j { 1.0 } else { 0.0 };
                    if (dot - want).abs() > 1e-6 {
                        return Err(invalid("SVD components are not orthonormal"));
                    }
                }
            }
            let flat: Vec<F> = rows
                .iter()
                .flat_map(|row| row.iter().map(|v| F::from_f64(*v)))
                .collect();
            let components = Dense::from_vec(flat, n_components, cols)?;
            Some(Fitted {
                components,
                singular_values,
                explained_variance: explained_variance.iter().map(|v| F::from_f64(*v)).collect(),
                cols,
            })
        } else {
            None
        };
        r.finish()?;
        Ok(Self {
            n_components,
            fitted,
            marker: core::marker::PhantomData,
        })
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
    fn full_rank_inverse_transform_recovers_the_data() {
        let data = [1.0_f64, 2.0, 3.0, 1.0, 0.5, 4.0];
        let mut svd = TruncatedSvd::<f64>::new(2);
        let z = svd
            .fit_transform(DenseView::new(&data, 3, 2).unwrap())
            .unwrap();
        let back = svd.inverse_transform(&z.view()).unwrap();
        for (a, b) in back.into_vec().iter().zip(&data) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn explained_variance_is_the_variance_of_the_projection() {
        let data = [1.0_f64, 2.0, 3.0, 1.0, 0.5, 4.0, 2.0, 2.0];
        let mut svd = TruncatedSvd::<f64>::new(2);
        let z = svd
            .fit_transform(DenseView::new(&data, 4, 2).unwrap())
            .unwrap();
        let ev = svd.explained_variance().unwrap();
        for (c, &variance) in ev.iter().enumerate() {
            let col: Vec<f64> = (0..4).map(|r| z.get(r, c)).collect();
            let mean = col.iter().sum::<f64>() / 4.0;
            let var = col.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 4.0;
            assert!((variance - var).abs() < 1e-9);
        }
        // Variances are never negative. They need not be ordered, since the singular values
        // rank the uncentred sums of squares, not these centred variances.
        assert!(ev.iter().all(|v| *v >= 0.0));
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

#[cfg(test)]
mod persist_tests {
    use super::*;
    use tpt_ml_proven_core::{DenseView, Transformer, UnsupervisedEstimator};

    fn data() -> Vec<f64> {
        (0..30).map(|i| ((i * 7) % 13) as f64 - 6.0).collect()
    }

    #[test]
    fn fitted_svd_round_trips_and_transforms_identically() {
        let x = data();
        let mut m = TruncatedSvd::<f64>::new(2);
        UnsupervisedEstimator::fit(&mut m, DenseView::new(&x, 10, 3).unwrap()).unwrap();
        let loaded = TruncatedSvd::<f64>::from_bytes(&m.to_bytes().unwrap()).unwrap();
        assert_eq!(loaded, m);
        let a = Transformer::transform(&m, DenseView::new(&x, 10, 3).unwrap()).unwrap();
        let b = Transformer::transform(&loaded, DenseView::new(&x, 10, 3).unwrap()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn unfitted_svd_round_trips() {
        let m = TruncatedSvd::<f32>::new(2);
        assert_eq!(
            TruncatedSvd::<f32>::from_bytes(&m.to_bytes().unwrap()).unwrap(),
            m
        );
    }

    #[test]
    fn f32_svd_keeps_its_dtype_tag() {
        let x: Vec<f32> = data().iter().map(|v| *v as f32).collect();
        let mut m = TruncatedSvd::<f32>::new(1);
        UnsupervisedEstimator::fit(&mut m, DenseView::new(&x, 10, 3).unwrap()).unwrap();
        let bytes = m.to_bytes().unwrap();
        let r = tpt_ml_proven_core::persist::Reader::new(&bytes, kind::TRUNCATED_SVD).unwrap();
        assert_eq!(r.dtype(), tpt_ml_proven_core::persist::dtype::F32);
        assert_eq!(TruncatedSvd::<f32>::from_bytes(&bytes).unwrap(), m);
    }
}
