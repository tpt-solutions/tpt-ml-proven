// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Model files for the regression models, in the format described by
//! `tpt_ml_proven_core::persist`. Hyper-parameters are stored with the fitted state, so a
//! loaded model can be refitted with the same settings.

use tpt_ml_proven_core::error::Error;
use tpt_ml_proven_core::persist::{kind, Reader, Writer};
use tpt_ml_proven_core::Float;

use crate::{ElasticNet, Lasso, Ridge};

fn invalid(reason: &'static str) -> Error {
    Error::Format { reason }
}

// ---------------------------------------------------------------- Ridge

impl<F: Float> Ridge<F> {
    /// Serialises the model (parameter and fitted state) to a model file.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut w = Writer::new(kind::RIDGE);
        w.f64(self.alpha);
        match &self.fitted {
            None => w.bool(false),
            Some((coef, intercept)) => {
                w.bool(true);
                w.f64s(coef);
                w.f64(*intercept);
            }
        }
        Ok(w.finish())
    }

    /// Loads a model written by [`Ridge::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(bytes, kind::RIDGE)?;
        let alpha = r.f64()?;
        let fitted = if r.bool()? {
            let coef = r.f64s()?;
            Some((coef, r.f64()?))
        } else {
            None
        };
        r.finish()?;
        if !(alpha.is_finite() && alpha >= 0.0) {
            return Err(invalid("ridge penalty is not finite and non-negative"));
        }
        Ok(Self {
            alpha,
            fitted,
            marker: core::marker::PhantomData,
        })
    }
}

// ---------------------------------------------------------------- ElasticNet and Lasso

/// Shared layout of the coordinate-descent models: alpha, l1_ratio, max_iter, tol, then
/// the fitted state (coefficients, intercept, sweeps used).
fn write_elastic_net<F: Float>(w: &mut Writer, m: &ElasticNet<F>) {
    w.f64(m.alpha);
    w.f64(m.l1_ratio);
    w.usize(m.max_iter);
    w.f64(m.tol);
    match &m.fitted {
        None => w.bool(false),
        Some((coef, intercept, sweeps)) => {
            w.bool(true);
            w.f64s(coef);
            w.f64(*intercept);
            w.usize(*sweeps);
        }
    }
}

fn read_elastic_net<F: Float>(r: &mut Reader<'_>) -> Result<ElasticNet<F>, Error> {
    let alpha = r.f64()?;
    let l1_ratio = r.f64()?;
    let max_iter = r.usize()?;
    let tol = r.f64()?;
    let fitted = if r.bool()? {
        let coef = r.f64s()?;
        let intercept = r.f64()?;
        Some((coef, intercept, r.usize()?))
    } else {
        None
    };
    if !(alpha.is_finite() && alpha >= 0.0) {
        return Err(invalid(
            "elastic net penalty is not finite and non-negative",
        ));
    }
    if !(0.0..=1.0).contains(&l1_ratio) {
        return Err(invalid("elastic net l1_ratio is outside [0, 1]"));
    }
    if !tol.is_finite() {
        return Err(invalid("elastic net tolerance is not finite"));
    }
    Ok(ElasticNet {
        alpha,
        l1_ratio,
        max_iter,
        tol,
        fitted,
        marker: core::marker::PhantomData,
    })
}

impl<F: Float> ElasticNet<F> {
    /// Serialises the model to a model file.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut w = Writer::new(kind::ELASTIC_NET);
        write_elastic_net(&mut w, self);
        Ok(w.finish())
    }

    /// Loads a model written by [`ElasticNet::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(bytes, kind::ELASTIC_NET)?;
        let m = read_elastic_net(&mut r)?;
        r.finish()?;
        Ok(m)
    }
}

impl<F: Float> Lasso<F> {
    /// Serialises the model to a model file.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut w = Writer::new(kind::LASSO);
        write_elastic_net(&mut w, &self.0);
        Ok(w.finish())
    }

    /// Loads a model written by [`Lasso::to_bytes`]. The file must describe a pure L1 model.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(bytes, kind::LASSO)?;
        let m = read_elastic_net(&mut r)?;
        r.finish()?;
        if m.l1_ratio != 1.0 {
            return Err(invalid("lasso model file has l1_ratio other than 1"));
        }
        Ok(Lasso(m))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ml_proven_core::{DenseView, Predictor, SupervisedEstimator};

    fn data() -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..24).map(|i| ((i * 7) % 11) as f64 - 5.0).collect();
        let y: Vec<f64> = (0..12)
            .map(|r| 2.0 * x[r * 2] - x[r * 2 + 1] + 0.5)
            .collect();
        (x, y)
    }

    #[test]
    fn ridge_round_trip_predicts_identically() {
        let (x, y) = data();
        let mut m = Ridge::<f64>::new(0.3);
        SupervisedEstimator::fit(&mut m, DenseView::new(&x, 12, 2).unwrap(), &y[..]).unwrap();
        let loaded = Ridge::<f64>::from_bytes(&m.to_bytes().unwrap()).unwrap();
        assert_eq!(m, loaded);
        let a = Predictor::predict(&m, DenseView::new(&x, 12, 2).unwrap()).unwrap();
        let b = Predictor::predict(&loaded, DenseView::new(&x, 12, 2).unwrap()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn unfitted_model_round_trips_too() {
        let m = Ridge::<f64>::new(1.0);
        assert_eq!(Ridge::<f64>::from_bytes(&m.to_bytes().unwrap()).unwrap(), m);
    }

    #[test]
    fn lasso_and_elastic_net_round_trip() {
        let (x, y) = data();
        let mut l = Lasso::<f64>::new(0.05);
        SupervisedEstimator::fit(&mut l, DenseView::new(&x, 12, 2).unwrap(), &y[..]).unwrap();
        let loaded = Lasso::<f64>::from_bytes(&l.to_bytes().unwrap()).unwrap();
        assert_eq!(loaded.coef(), l.coef());
        assert_eq!(loaded.intercept(), l.intercept());

        let mut e = ElasticNet::<f64>::new(0.05, 0.4);
        SupervisedEstimator::fit(&mut e, DenseView::new(&x, 12, 2).unwrap(), &y[..]).unwrap();
        assert_eq!(
            ElasticNet::<f64>::from_bytes(&e.to_bytes().unwrap()).unwrap(),
            e
        );
    }

    #[test]
    fn a_ridge_file_is_not_a_lasso() {
        let bytes = Ridge::<f64>::new(1.0).to_bytes().unwrap();
        assert!(matches!(
            Lasso::<f64>::from_bytes(&bytes),
            Err(Error::Format { .. })
        ));
    }
}
