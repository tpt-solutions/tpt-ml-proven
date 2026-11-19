// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Missing-value imputation. A missing value is NaN. Unlike the other estimators, the
//! imputer accepts NaN in its input, since that is what it is for.

use tpt_ml_proven_core::error::{Error, ShapeError};
use tpt_ml_proven_core::{ensures, ArrayView, Dense, Float, Transformer, UnsupervisedEstimator};

/// How the fill value for each column is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImputeStrategy {
    /// The mean of the observed values.
    Mean,
    /// The median of the observed values.
    Median,
    /// The most frequent observed value. Ties go to the smallest value.
    MostFrequent,
    /// A fixed value.
    Constant(f64),
}

/// Fills NaN entries with a per-column statistic learned in `fit`.
#[derive(Debug, Clone, PartialEq)]
pub struct SimpleImputer<F: Float> {
    strategy: ImputeStrategy,
    fill: Option<Vec<f64>>,
    marker: core::marker::PhantomData<F>,
}

impl<F: Float> SimpleImputer<F> {
    /// An unfitted imputer with the given strategy.
    pub fn new(strategy: ImputeStrategy) -> Self {
        Self {
            strategy,
            fill: None,
            marker: core::marker::PhantomData,
        }
    }

    /// Learned fill value for each column.
    pub fn statistics(&self) -> Option<&[f64]> {
        self.fill.as_deref()
    }
}

fn observed<F: Float, X: ArrayView<Elem = F>>(x: &X, c: usize) -> Vec<f64> {
    (0..x.rows())
        .map(|r| x.get(r, c).to_f64())
        .filter(|v| !v.is_nan())
        .collect()
}

fn statistic(strategy: ImputeStrategy, mut vals: Vec<f64>) -> Option<f64> {
    if let ImputeStrategy::Constant(v) = strategy {
        return Some(v);
    }
    if vals.is_empty() {
        return None;
    }
    match strategy {
        ImputeStrategy::Mean => Some(vals.iter().sum::<f64>() / vals.len() as f64),
        ImputeStrategy::Median => {
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            let n = vals.len();
            Some(if n % 2 == 1 {
                vals[n / 2]
            } else {
                (vals[n / 2 - 1] + vals[n / 2]) / 2.0
            })
        }
        ImputeStrategy::MostFrequent => {
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            let (mut best, mut best_count) = (vals[0], 0);
            let mut i = 0;
            while i < vals.len() {
                let mut j = i;
                while j < vals.len() && vals[j] == vals[i] {
                    j += 1;
                }
                if j - i > best_count {
                    best = vals[i];
                    best_count = j - i;
                }
                i = j;
            }
            Some(best)
        }
        ImputeStrategy::Constant(_) => unreachable!("handled above"),
    }
}

impl<F: Float> UnsupervisedEstimator for SimpleImputer<F> {
    type Float = F;
    type FitError = Error;

    fn fit<X>(&mut self, x: X) -> Result<(), Error>
    where
        X: ArrayView<Elem = F>,
    {
        self.fill = None;
        if x.rows() == 0 || x.cols() == 0 {
            return Err(Error::EmptyInput);
        }
        let mut fill = Vec::with_capacity(x.cols());
        for c in 0..x.cols() {
            let v = statistic(self.strategy, observed(&x, c)).ok_or(Error::InvalidParameter {
                name: "X",
                reason: "a column has no observed values to learn a fill value from",
            })?;
            fill.push(v);
        }
        self.fill = Some(fill);
        Ok(())
    }
}

impl<F: Float> Transformer for SimpleImputer<F> {
    type Float = F;
    type Output = Dense<F>;

    fn transform<X>(&self, x: X) -> Result<Dense<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        let fill = self.fill.as_ref().ok_or(Error::NotFitted)?;
        if x.cols() != fill.len() {
            return Err(Error::Shape(ShapeError {
                rows: x.rows(),
                cols: x.cols(),
                reason: "feature count differs from the fitted imputer",
            }));
        }
        let mut out = Vec::with_capacity(x.rows() * x.cols());
        for r in 0..x.rows() {
            for (c, &f) in fill.iter().enumerate() {
                let v = x.get(r, c);
                out.push(if v.is_finite() { v } else { F::from_f64(f) });
            }
        }
        // Postcondition: no missing value survives the imputation.
        ensures!(
            out.iter().all(|v| v.is_finite()),
            "simple_imputer.no_missing"
        );
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

// ---------------------------------------------------------------- model files

use tpt_ml_proven_core::persist::{kind, Reader, Writer};

fn invalid(reason: &'static str) -> Error {
    Error::Format { reason }
}

impl<F: Float> SimpleImputer<F> {
    /// Serialises the imputer (strategy and learned fill values) to a model file.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut w = Writer::for_float::<F>(kind::SIMPLE_IMPUTER);
        match self.strategy {
            ImputeStrategy::Mean => w.usize(0),
            ImputeStrategy::Median => w.usize(1),
            ImputeStrategy::MostFrequent => w.usize(2),
            ImputeStrategy::Constant(_) => w.usize(3),
        }
        // Always written, so the layout does not depend on the strategy.
        w.f64(match self.strategy {
            ImputeStrategy::Constant(v) => v,
            _ => 0.0,
        });
        match &self.fill {
            None => w.bool(false),
            Some(fill) => {
                w.bool(true);
                w.f64s(fill);
            }
        }
        Ok(w.finish())
    }

    /// Loads an imputer written by [`SimpleImputer::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(bytes, kind::SIMPLE_IMPUTER)?;
        let tag = r.usize()?;
        let constant = r.f64()?;
        let strategy = match tag {
            0 => ImputeStrategy::Mean,
            1 => ImputeStrategy::Median,
            2 => ImputeStrategy::MostFrequent,
            3 => ImputeStrategy::Constant(constant),
            _ => return Err(invalid("unknown imputation strategy")),
        };
        let fill = if r.bool()? { Some(r.f64s()?) } else { None };
        r.finish()?;
        if let Some(f) = &fill {
            if f.is_empty() {
                return Err(invalid("imputer has no fill values"));
            }
        }
        Ok(Self {
            strategy,
            fill,
            marker: core::marker::PhantomData,
        })
    }
}

impl<F: Float> Default for SimpleImputer<F> {
    /// The default settings, as documented on the constructor.
    fn default() -> Self {
        Self::new(ImputeStrategy::Mean)
    }
}

// ---------------------------------------------------------------- settings

impl<F: Float> SimpleImputer<F> {
    /// The strategy the imputer was configured with.
    pub fn strategy(&self) -> ImputeStrategy {
        self.strategy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ml_proven_core::DenseView;

    const NAN: f64 = f64::NAN;

    fn impute(strategy: ImputeStrategy, data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        let mut imp = SimpleImputer::new(strategy);
        imp.fit_transform(DenseView::new(data, rows, cols).unwrap())
            .unwrap()
            .into_vec()
    }

    #[test]
    fn mean_and_median_fill_observed_statistics() {
        let data = [1.0_f64, NAN, 3.0, 10.0, NAN, 30.0];
        assert_eq!(
            impute(ImputeStrategy::Mean, &data, 3, 2),
            vec![1.0, 20.0, 3.0, 10.0, 2.0, 30.0]
        );
        let data = [1.0_f64, NAN, 2.0, 10.0, 9.0, 30.0];
        assert_eq!(
            impute(ImputeStrategy::Median, &data, 3, 2),
            vec![1.0, 20.0, 2.0, 10.0, 9.0, 30.0]
        );
    }

    #[test]
    fn most_frequent_breaks_ties_towards_the_smallest() {
        // Column 0 is [2, NaN, 1]: the observed values tie at 1 and 2, so the fill is 1.
        let data = [2.0_f64, 2.0, NAN, 1.0, 1.0, 3.0];
        let out = impute(ImputeStrategy::MostFrequent, &data, 3, 2);
        assert_eq!(out[2], 1.0);
    }

    #[test]
    fn constant_fills_without_looking_at_data() {
        let data = [NAN, 5.0, 2.0, NAN];
        assert_eq!(
            impute(ImputeStrategy::Constant(-1.0), &data, 2, 2),
            vec![-1.0, 5.0, 2.0, -1.0]
        );
    }

    #[test]
    fn all_missing_column_is_an_error() {
        let data = [NAN, 1.0, NAN, 2.0];
        let mut imp = SimpleImputer::new(ImputeStrategy::Mean);
        assert!(matches!(
            UnsupervisedEstimator::fit(&mut imp, DenseView::new(&data, 2, 2).unwrap()),
            Err(Error::InvalidParameter { .. })
        ));
    }
}

#[cfg(test)]
mod persist_tests {
    use super::*;
    use tpt_ml_proven_core::{DenseView, Transformer, UnsupervisedEstimator};

    #[test]
    fn every_strategy_round_trips_with_its_fill_values() {
        let x = vec![1.0, f64::NAN, 3.0, 4.0, 5.0, 4.0, f64::NAN, 2.0];
        for strategy in [
            ImputeStrategy::Mean,
            ImputeStrategy::Median,
            ImputeStrategy::MostFrequent,
            ImputeStrategy::Constant(-9.0),
        ] {
            let mut imp = SimpleImputer::<f64>::new(strategy);
            UnsupervisedEstimator::fit(&mut imp, DenseView::new(&x, 4, 2).unwrap()).unwrap();
            let loaded = SimpleImputer::<f64>::from_bytes(&imp.to_bytes().unwrap()).unwrap();
            assert_eq!(loaded, imp);
            let a = Transformer::transform(&imp, DenseView::new(&x, 4, 2).unwrap()).unwrap();
            let b = Transformer::transform(&loaded, DenseView::new(&x, 4, 2).unwrap()).unwrap();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn unfitted_imputer_round_trips() {
        let imp = SimpleImputer::<f64>::new(ImputeStrategy::Mean);
        assert_eq!(
            SimpleImputer::<f64>::from_bytes(&imp.to_bytes().unwrap()).unwrap(),
            imp
        );
    }
}
