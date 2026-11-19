// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Encoders for categorical columns. Categories are numbers; each column's categories
//! are its distinct values, in increasing order. Unknown categories at transform time
//! are an error, as in scikit-learn's default `handle_unknown="error"`.

use tpt_ml_proven_core::error::{Error, ShapeError};
use tpt_ml_proven_core::{ensures, ArrayView, Dense, Float, Transformer, UnsupervisedEstimator};

/// Sorted distinct values of each column.
fn categories<F: Float, X: ArrayView<Elem = F>>(x: &X) -> Result<Vec<Vec<f64>>, Error> {
    if x.rows() == 0 || x.cols() == 0 {
        return Err(Error::EmptyInput);
    }
    if !x.all_finite() {
        return Err(Error::NonFinite { what: "X" });
    }
    let mut out = Vec::with_capacity(x.cols());
    for c in 0..x.cols() {
        let mut vals: Vec<f64> = (0..x.rows()).map(|r| x.get(r, c).to_f64()).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        vals.dedup();
        out.push(vals);
    }
    Ok(out)
}

fn position(cats: &[f64], v: f64) -> Option<usize> {
    cats.binary_search_by(|c| c.partial_cmp(&v).unwrap_or(core::cmp::Ordering::Equal))
        .ok()
}

fn unknown(v: f64) -> Error {
    Error::InvalidParameter {
        name: "X",
        reason: if v.is_nan() {
            "unknown category NaN"
        } else {
            "unknown category at transform time"
        },
    }
}

/// Encodes each category as a 0/1 column. The output has one column per category of
/// every input column, in input order.
#[derive(Debug, Clone, PartialEq)]
pub struct OneHotEncoder<F: Float> {
    categories: Option<Vec<Vec<f64>>>,
    marker: core::marker::PhantomData<F>,
}

impl<F: Float> OneHotEncoder<F> {
    /// An unfitted encoder.
    pub fn new() -> Self {
        Self {
            categories: None,
            marker: core::marker::PhantomData,
        }
    }

    /// Learned categories per input column.
    pub fn categories(&self) -> Option<&[Vec<f64>]> {
        self.categories.as_deref()
    }
}

impl<F: Float> UnsupervisedEstimator for OneHotEncoder<F> {
    type Float = F;
    type FitError = Error;

    fn fit<X>(&mut self, x: X) -> Result<(), Error>
    where
        X: ArrayView<Elem = F>,
    {
        self.categories = None;
        self.categories = Some(categories(&x)?);
        Ok(())
    }
}

impl<F: Float> Transformer for OneHotEncoder<F> {
    type Float = F;
    type Output = Dense<F>;

    fn transform<X>(&self, x: X) -> Result<Dense<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        let cats = self.categories.as_ref().ok_or(Error::NotFitted)?;
        if x.cols() != cats.len() {
            return Err(Error::Shape(ShapeError {
                rows: x.rows(),
                cols: x.cols(),
                reason: "feature count differs from the fitted encoder",
            }));
        }
        let width: usize = cats.iter().map(Vec::len).sum();
        let mut out = vec![F::ZERO; x.rows() * width];
        for r in 0..x.rows() {
            let mut offset = 0;
            for (c, col_cats) in cats.iter().enumerate() {
                let v = x.get(r, c).to_f64();
                let k = position(col_cats, v).ok_or_else(|| unknown(v))?;
                out[r * width + offset + k] = F::ONE;
                offset += col_cats.len();
            }
        }
        // Postcondition: each input feature sets exactly one indicator, so every row
        // contributes one hot entry per feature.
        let hot_per_row = out.iter().filter(|v| **v == F::ONE).count();
        ensures!(
            hot_per_row == x.rows() * cats.len(),
            "one_hot.one_per_feature"
        );
        Ok(Dense::from_vec(out, x.rows(), width)?)
    }

    fn fit_transform<X>(&mut self, x: X) -> Result<Dense<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        UnsupervisedEstimator::fit(self, &x)?;
        self.transform(x)
    }
}

/// Encodes each category as its position in the sorted categories of its column.
#[derive(Debug, Clone, PartialEq)]
pub struct OrdinalEncoder<F: Float> {
    categories: Option<Vec<Vec<f64>>>,
    marker: core::marker::PhantomData<F>,
}

impl<F: Float> OrdinalEncoder<F> {
    /// An unfitted encoder.
    pub fn new() -> Self {
        Self {
            categories: None,
            marker: core::marker::PhantomData,
        }
    }

    /// Learned categories per input column.
    pub fn categories(&self) -> Option<&[Vec<f64>]> {
        self.categories.as_deref()
    }
}

impl<F: Float> UnsupervisedEstimator for OrdinalEncoder<F> {
    type Float = F;
    type FitError = Error;

    fn fit<X>(&mut self, x: X) -> Result<(), Error>
    where
        X: ArrayView<Elem = F>,
    {
        self.categories = None;
        self.categories = Some(categories(&x)?);
        Ok(())
    }
}

impl<F: Float> Transformer for OrdinalEncoder<F> {
    type Float = F;
    type Output = Dense<F>;

    fn transform<X>(&self, x: X) -> Result<Dense<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        let cats = self.categories.as_ref().ok_or(Error::NotFitted)?;
        if x.cols() != cats.len() {
            return Err(Error::Shape(ShapeError {
                rows: x.rows(),
                cols: x.cols(),
                reason: "feature count differs from the fitted encoder",
            }));
        }
        let mut out = Vec::with_capacity(x.rows() * x.cols());
        for r in 0..x.rows() {
            for (c, col_cats) in cats.iter().enumerate() {
                let v = x.get(r, c).to_f64();
                let k = position(col_cats, v).ok_or_else(|| unknown(v))?;
                // Round trip: the code points back at the original value.
                ensures!(col_cats[k] == v, "ordinal.round_trip");
                out.push(F::from_usize(k));
            }
        }
        Ok(Dense::from_vec(out, x.rows(), x.cols())?)
    }

    fn fit_transform<X>(&mut self, x: X) -> Result<Dense<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        UnsupervisedEstimator::fit(self, &x)?;
        self.transform(x)
    }
}

impl<F: Float> Default for OneHotEncoder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> Default for OrdinalEncoder<F> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ml_proven_core::DenseView;

    #[test]
    fn one_hot_encodes_sorted_categories() {
        // Column 0 categories [1, 3]; column 1 categories [10, 20].
        let data = [3.0_f64, 10.0, 1.0, 20.0, 3.0, 10.0];
        let mut e = OneHotEncoder::<f64>::new();
        let out = e
            .fit_transform(DenseView::new(&data, 3, 2).unwrap())
            .unwrap();
        assert_eq!(out.cols(), 4);
        assert_eq!(
            out.into_vec(),
            vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
        );
    }

    #[test]
    fn ordinal_encodes_positions() {
        let data = [3.0_f64, 10.0, 1.0, 20.0, 3.0, 10.0];
        let mut e = OrdinalEncoder::<f64>::new();
        let out = e
            .fit_transform(DenseView::new(&data, 3, 2).unwrap())
            .unwrap();
        assert_eq!(out.into_vec(), vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn unknown_category_is_an_error() {
        let data = [1.0_f64, 2.0];
        let mut e = OrdinalEncoder::<f64>::new();
        UnsupervisedEstimator::fit(&mut e, DenseView::new(&data, 2, 1).unwrap()).unwrap();
        let unseen = [9.0_f64];
        assert!(matches!(
            e.transform(DenseView::new(&unseen, 1, 1).unwrap()),
            Err(Error::InvalidParameter { .. })
        ));
    }
}
