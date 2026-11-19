// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! A transformer followed by an estimator, as a plain generic struct.
//!
//! There is no runtime pipeline object: `Pipeline<T, E>` is a concrete type whose methods are
//! monomorphised, so composition costs nothing at run time. Build one with
//! [`Pipeline::new`], or with the [`Then::then`] helper on any transformer:
//!
//! ```
//! use tpt_ml_proven_core::toy::{ColumnCenterer, MeanRegressor};
//! use tpt_ml_proven_core::{DenseView, Predictor, SupervisedEstimator, Then};
//!
//! let x = [1.0_f64, 10.0, 3.0, 30.0];
//! let y = [2.0_f64, 4.0];
//! let mut model = ColumnCenterer::<f64>::new().then(MeanRegressor::<f64>::new());
//! model.fit(DenseView::new(&x, 2, 2).unwrap(), &y[..]).unwrap();
//! let pred = model.predict(DenseView::new(&x, 2, 2).unwrap()).unwrap();
//! assert_eq!(pred, vec![3.0, 3.0]);
//! ```

use crate::error::Error;
use crate::{
    ArrayView, Dense, Float, Predictor, SupervisedEstimator, TargetView, Transformer,
    UnsupervisedEstimator,
};

/// A transformer `T` whose output feeds an estimator `E`. Fitting learns the transformer
/// first, then the estimator on the transformed data. Predicting applies both in order.
#[derive(Debug, Clone, PartialEq)]
pub struct Pipeline<T, E> {
    /// The first stage.
    pub transform: T,
    /// The final stage.
    pub estimator: E,
}

impl<T, E> Pipeline<T, E> {
    /// Chains `transform` into `estimator`.
    pub fn new(transform: T, estimator: E) -> Self {
        Self {
            transform,
            estimator,
        }
    }
}

impl<F, T, E> SupervisedEstimator for Pipeline<T, E>
where
    F: Float,
    T: UnsupervisedEstimator<Float = F, FitError = Error>
        + Transformer<Float = F, Output = Dense<F>>,
    E: SupervisedEstimator<Float = F, FitError = Error>,
{
    type Float = F;
    type FitError = Error;

    fn fit<X, Y>(&mut self, x: X, y: Y) -> Result<(), Error>
    where
        X: ArrayView<Elem = F>,
        Y: TargetView<Elem = F>,
    {
        UnsupervisedEstimator::fit(&mut self.transform, &x)?;
        let z = Transformer::transform(&self.transform, x)?;
        SupervisedEstimator::fit(&mut self.estimator, z.view(), y)
    }
}

impl<F, T, E> Predictor for Pipeline<T, E>
where
    F: Float,
    T: Transformer<Float = F, Output = Dense<F>>,
    E: Predictor<Float = F>,
{
    type Float = F;
    type Output = E::Output;

    fn predict<X>(&self, x: X) -> Result<Self::Output, Error>
    where
        X: ArrayView<Elem = F>,
    {
        let z = Transformer::transform(&self.transform, x)?;
        Predictor::predict(&self.estimator, z.view())
    }
}

/// Adds `.then(estimator)` to every transformer, to build a [`Pipeline`].
pub trait Then: Sized {
    /// Chains this transformer into `estimator`.
    fn then<E>(self, estimator: E) -> Pipeline<Self, E> {
        Pipeline::new(self, estimator)
    }
}

impl<T: Transformer> Then for T {}
