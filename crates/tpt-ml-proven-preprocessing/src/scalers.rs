// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Per-feature scalers. Each one learns, per column, an affine map `y = x * factor +
//! offset` and applies it. The inverse is `x = (y - offset) / factor`.

use tpt_ml_proven_core::error::{Error, ShapeError};
use tpt_ml_proven_core::{ensures, ArrayView, Dense, Float, Transformer, UnsupervisedEstimator};

use crate::nonzero;

/// Relative slack for the scaler postconditions, in floating point.
const SCALER_TOL: f64 = 1e-8;

/// Learned per-column affine map.
#[derive(Debug, Clone, PartialEq)]
struct Affine {
    offset: Vec<f64>,
    factor: Vec<f64>,
}

impl Affine {
    fn apply<F: Float, X: ArrayView<Elem = F>>(&self, x: &X) -> Result<Dense<F>, Error> {
        check_cols(x, self.offset.len())?;
        let mut out = Vec::with_capacity(x.rows() * x.cols());
        for r in 0..x.rows() {
            for c in 0..x.cols() {
                let v = x.get(r, c).to_f64() * self.factor[c] + self.offset[c];
                out.push(F::from_f64(v));
            }
        }
        Ok(Dense::from_vec(out, x.rows(), x.cols())?)
    }

    fn invert<F: Float, X: ArrayView<Elem = F>>(&self, y: &X) -> Result<Dense<F>, Error> {
        check_cols(y, self.offset.len())?;
        let mut out = Vec::with_capacity(y.rows() * y.cols());
        for r in 0..y.rows() {
            for c in 0..y.cols() {
                let v = (y.get(r, c).to_f64() - self.offset[c]) / self.factor[c];
                out.push(F::from_f64(v));
            }
        }
        Ok(Dense::from_vec(out, y.rows(), y.cols())?)
    }
}

fn check_cols<X: ArrayView>(x: &X, expected: usize) -> Result<(), Error> {
    if x.cols() != expected {
        return Err(Error::Shape(ShapeError {
            rows: x.rows(),
            cols: x.cols(),
            reason: "feature count differs from the fitted scaler",
        }));
    }
    Ok(())
}

/// Column `c` of a fitted-on matrix, as `f64`, for statistics.
fn column<F: Float, X: ArrayView<Elem = F>>(x: &X, c: usize) -> Vec<f64> {
    (0..x.rows()).map(|r| x.get(r, c).to_f64()).collect()
}

fn validate<F: Float, X: ArrayView<Elem = F>>(x: &X) -> Result<(), Error> {
    if x.rows() == 0 || x.cols() == 0 {
        return Err(Error::EmptyInput);
    }
    if !x.all_finite() {
        return Err(Error::NonFinite { what: "X" });
    }
    Ok(())
}

/// Linear-interpolation quantile, as `numpy.percentile` computes it.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (pos - lo as f64) * (sorted[hi] - sorted[lo])
}

// ---------------------------------------------------------------- StandardScaler

/// Standardises each feature to zero mean and unit (population) variance.
#[derive(Debug, Clone, PartialEq)]
pub struct StandardScaler<F: Float> {
    fitted: Option<Affine>,
    mean: Vec<f64>,
    scale: Vec<f64>,
    marker: core::marker::PhantomData<F>,
}

impl<F: Float> StandardScaler<F> {
    /// An unfitted scaler.
    pub fn new() -> Self {
        Self {
            fitted: None,
            mean: Vec::new(),
            scale: Vec::new(),
            marker: core::marker::PhantomData,
        }
    }

    /// Learned per-feature means.
    pub fn mean(&self) -> Option<&[f64]> {
        self.fitted.as_ref().map(|_| self.mean.as_slice())
    }

    /// Learned per-feature standard deviations (zero replaced by one).
    pub fn scale(&self) -> Option<&[f64]> {
        self.fitted.as_ref().map(|_| self.scale.as_slice())
    }

    /// Maps standardised values back to the original scale.
    pub fn inverse_transform<G: Float, Y: ArrayView<Elem = G>>(
        &self,
        y: &Y,
    ) -> Result<Dense<G>, Error> {
        self.fitted.as_ref().ok_or(Error::NotFitted)?.invert(y)
    }
}

impl<F: Float> UnsupervisedEstimator for StandardScaler<F> {
    type Float = F;
    type FitError = Error;

    fn fit<X>(&mut self, x: X) -> Result<(), Error>
    where
        X: ArrayView<Elem = F>,
    {
        self.fitted = None;
        validate(&x)?;
        let n = x.rows() as f64;
        let mut mean = Vec::new();
        let mut scale = Vec::new();
        // Standard deviation before the constant-column fallback; zero marks a constant column.
        let mut raw_std = Vec::new();
        for c in 0..x.cols() {
            let col = column(&x, c);
            let m = col.iter().sum::<f64>() / n;
            let var = col.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n;
            mean.push(m);
            raw_std.push(var.sqrt());
            scale.push(nonzero(var.sqrt()));
        }
        let factor: Vec<f64> = scale.iter().map(|s| 1.0 / s).collect();
        let offset: Vec<f64> = mean.iter().zip(&scale).map(|(m, s)| -m / s).collect();

        // Postconditions on the training data: every column is centred, and every
        // non-constant column has unit variance. Constant columns map to zero.
        let (mut worst_mean, mut worst_var, mut largest) = (0.0_f64, 0.0_f64, 1.0_f64);
        let mut z_cols = Vec::with_capacity(x.cols());
        for c in 0..x.cols() {
            let z: Vec<f64> = column(&x, c)
                .iter()
                .map(|v| v * factor[c] + offset[c])
                .collect();
            largest = largest.max(z.iter().fold(0.0_f64, |a, v| a.max(v.abs())));
            z_cols.push(z);
        }
        for (c, z) in z_cols.iter().enumerate() {
            let zm = z.iter().sum::<f64>() / n;
            let zv = z.iter().map(|v| (v - zm) * (v - zm)).sum::<f64>() / n;
            worst_mean = worst_mean.max(zm.abs());
            if raw_std[c] > 0.0 {
                worst_var = worst_var.max((zv - 1.0).abs());
            }
        }
        ensures!(
            worst_mean <= SCALER_TOL * largest,
            "standard_scaler.centred"
        );
        ensures!(
            worst_var <= SCALER_TOL * largest,
            "standard_scaler.unit_variance"
        );

        self.mean = mean;
        self.scale = scale;
        self.fitted = Some(Affine { offset, factor });
        Ok(())
    }
}

impl<F: Float> Transformer for StandardScaler<F> {
    type Float = F;
    type Output = Dense<F>;

    fn transform<X>(&self, x: X) -> Result<Dense<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        self.fitted.as_ref().ok_or(Error::NotFitted)?.apply(&x)
    }

    fn fit_transform<X>(&mut self, x: X) -> Result<Dense<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        UnsupervisedEstimator::fit(self, &x)?;
        self.transform(x)
    }
}

// ---------------------------------------------------------------- MinMaxScaler

/// Rescales each feature to `[0, 1]` (the default range).
#[derive(Debug, Clone, PartialEq)]
pub struct MinMaxScaler<F: Float> {
    fitted: Option<Affine>,
    marker: core::marker::PhantomData<F>,
}

impl<F: Float> MinMaxScaler<F> {
    /// An unfitted scaler with range `[0, 1]`.
    pub fn new() -> Self {
        Self {
            fitted: None,
            marker: core::marker::PhantomData,
        }
    }

    /// Maps scaled values back to the original range.
    pub fn inverse_transform<G: Float, Y: ArrayView<Elem = G>>(
        &self,
        y: &Y,
    ) -> Result<Dense<G>, Error> {
        self.fitted.as_ref().ok_or(Error::NotFitted)?.invert(y)
    }
}

impl<F: Float> UnsupervisedEstimator for MinMaxScaler<F> {
    type Float = F;
    type FitError = Error;

    fn fit<X>(&mut self, x: X) -> Result<(), Error>
    where
        X: ArrayView<Elem = F>,
    {
        self.fitted = None;
        validate(&x)?;
        let mut factor = Vec::new();
        let mut offset = Vec::new();
        for c in 0..x.cols() {
            let col = column(&x, c);
            let lo = col.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let scale = 1.0 / nonzero(hi - lo);
            factor.push(scale);
            offset.push(-lo * scale);
        }
        // Postcondition: the training data lands in [0, 1] (up to rounding).
        let below = (0..x.cols()).fold(0.0_f64, |acc, c| {
            column(&x, c)
                .iter()
                .fold(acc, |a, v| a.max(-(v * factor[c] + offset[c])))
        });
        let above = (0..x.cols()).fold(0.0_f64, |acc, c| {
            column(&x, c)
                .iter()
                .fold(acc, |a, v| a.max(v * factor[c] + offset[c] - 1.0))
        });
        ensures!(below <= SCALER_TOL, "minmax_scaler.within_range");
        ensures!(above <= SCALER_TOL, "minmax_scaler.within_range");
        self.fitted = Some(Affine { offset, factor });
        Ok(())
    }
}

impl<F: Float> Transformer for MinMaxScaler<F> {
    type Float = F;
    type Output = Dense<F>;

    fn transform<X>(&self, x: X) -> Result<Dense<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        self.fitted.as_ref().ok_or(Error::NotFitted)?.apply(&x)
    }

    fn fit_transform<X>(&mut self, x: X) -> Result<Dense<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        UnsupervisedEstimator::fit(self, &x)?;
        self.transform(x)
    }
}

// ---------------------------------------------------------------- RobustScaler

/// Centres each feature on its median and scales by its interquartile range, so
/// outliers have little effect.
#[derive(Debug, Clone, PartialEq)]
pub struct RobustScaler<F: Float> {
    fitted: Option<Affine>,
    marker: core::marker::PhantomData<F>,
}

impl<F: Float> RobustScaler<F> {
    /// An unfitted scaler using the median and the 25th to 75th percentile range.
    pub fn new() -> Self {
        Self {
            fitted: None,
            marker: core::marker::PhantomData,
        }
    }

    /// Maps robust-scaled values back to the original scale.
    pub fn inverse_transform<G: Float, Y: ArrayView<Elem = G>>(
        &self,
        y: &Y,
    ) -> Result<Dense<G>, Error> {
        self.fitted.as_ref().ok_or(Error::NotFitted)?.invert(y)
    }
}

impl<F: Float> UnsupervisedEstimator for RobustScaler<F> {
    type Float = F;
    type FitError = Error;

    fn fit<X>(&mut self, x: X) -> Result<(), Error>
    where
        X: ArrayView<Elem = F>,
    {
        self.fitted = None;
        validate(&x)?;
        let mut factor = Vec::new();
        let mut offset = Vec::new();
        for c in 0..x.cols() {
            let mut col = column(&x, c);
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            let median = quantile(&col, 0.5);
            let iqr = quantile(&col, 0.75) - quantile(&col, 0.25);
            let scale = nonzero(iqr);
            factor.push(1.0 / scale);
            offset.push(-median / scale);
        }
        self.fitted = Some(Affine { offset, factor });
        Ok(())
    }
}

impl<F: Float> Transformer for RobustScaler<F> {
    type Float = F;
    type Output = Dense<F>;

    fn transform<X>(&self, x: X) -> Result<Dense<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        self.fitted.as_ref().ok_or(Error::NotFitted)?.apply(&x)
    }

    fn fit_transform<X>(&mut self, x: X) -> Result<Dense<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        UnsupervisedEstimator::fit(self, &x)?;
        self.transform(x)
    }
}

impl<F: Float> Default for StandardScaler<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> Default for MinMaxScaler<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> Default for RobustScaler<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------- model files

use tpt_ml_proven_core::persist::{kind, Reader, Writer};

fn invalid(reason: &'static str) -> Error {
    Error::Format { reason }
}

fn write_affine(w: &mut Writer, a: &Affine) {
    w.f64s(&a.offset);
    w.f64s(&a.factor);
}

fn read_affine(r: &mut Reader<'_>) -> Result<Affine, Error> {
    let offset = r.f64s()?;
    let factor = r.f64s()?;
    if offset.len() != factor.len() {
        return Err(invalid("scaler offset and factor lengths differ"));
    }
    if factor.iter().any(|&f| f <= 0.0) {
        return Err(invalid("scaler factor is not positive"));
    }
    Ok(Affine { offset, factor })
}

/// Shared load path for the scalers that hold only an affine map.
fn read_affine_model(bytes: &[u8], expected: u32) -> Result<Option<Affine>, Error> {
    let mut r = Reader::new(bytes, expected)?;
    let fitted = if r.bool()? {
        Some(read_affine(&mut r)?)
    } else {
        None
    };
    r.finish()?;
    Ok(fitted)
}

impl<F: Float> MinMaxScaler<F> {
    /// Serialises the fitted scaler to a model file.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut w = Writer::for_float::<F>(kind::MINMAX_SCALER);
        match &self.fitted {
            None => w.bool(false),
            Some(a) => {
                w.bool(true);
                write_affine(&mut w, a);
            }
        }
        Ok(w.finish())
    }

    /// Loads a scaler written by [`MinMaxScaler::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        Ok(Self {
            fitted: read_affine_model(bytes, kind::MINMAX_SCALER)?,
            marker: core::marker::PhantomData,
        })
    }
}

impl<F: Float> RobustScaler<F> {
    /// Serialises the fitted scaler to a model file.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut w = Writer::for_float::<F>(kind::ROBUST_SCALER);
        match &self.fitted {
            None => w.bool(false),
            Some(a) => {
                w.bool(true);
                write_affine(&mut w, a);
            }
        }
        Ok(w.finish())
    }

    /// Loads a scaler written by [`RobustScaler::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        Ok(Self {
            fitted: read_affine_model(bytes, kind::ROBUST_SCALER)?,
            marker: core::marker::PhantomData,
        })
    }
}

impl<F: Float> StandardScaler<F> {
    /// Serialises the fitted scaler to a model file.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut w = Writer::for_float::<F>(kind::STANDARD_SCALER);
        match &self.fitted {
            None => w.bool(false),
            Some(a) => {
                w.bool(true);
                write_affine(&mut w, a);
                w.f64s(&self.mean);
                w.f64s(&self.scale);
            }
        }
        Ok(w.finish())
    }

    /// Loads a scaler written by [`StandardScaler::to_bytes`]. The per-column mean and scale
    /// must agree with the affine map, which is `factor = 1 / scale`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(bytes, kind::STANDARD_SCALER)?;
        let (fitted, mean, scale) = if r.bool()? {
            let a = read_affine(&mut r)?;
            let mean = r.f64s()?;
            let scale = r.f64s()?;
            (Some(a), mean, scale)
        } else {
            (None, Vec::new(), Vec::new())
        };
        r.finish()?;
        if let Some(a) = &fitted {
            if mean.len() != a.factor.len() || scale.len() != a.factor.len() {
                return Err(invalid("standard scaler mean and scale lengths differ"));
            }
            // The map is x ↦ (x − mean) / scale, so factor = 1 / scale and offset = −mean / scale.
            for (i, (&f, &s)) in a.factor.iter().zip(&scale).enumerate() {
                let m = mean[i];
                let want_offset = -m / s;
                if s <= 0.0
                    || (f * s - 1.0).abs() > 1e-9
                    || (a.offset[i] - want_offset).abs() > 1e-9 * (1.0 + want_offset.abs())
                {
                    return Err(invalid(
                        "standard scaler mean and scale disagree with its map",
                    ));
                }
            }
        }
        Ok(Self {
            fitted,
            mean,
            scale,
            marker: core::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ml_proven_core::DenseView;

    fn view(data: &[f64], rows: usize, cols: usize) -> DenseView<'_, f64> {
        DenseView::new(data, rows, cols).unwrap()
    }

    #[test]
    fn standard_scaler_gives_zero_mean_unit_variance() {
        let data = [1.0_f64, 10.0, 2.0, 20.0, 3.0, 30.0];
        let mut s = StandardScaler::new();
        let out = s.fit_transform(view(&data, 3, 2)).unwrap();
        for c in 0..2 {
            let col: Vec<f64> = (0..3).map(|r| out.get(r, c)).collect();
            let mean = col.iter().sum::<f64>() / 3.0;
            let var = col.iter().map(|v| v * v).sum::<f64>() / 3.0;
            assert!(mean.abs() < 1e-12 && (var - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn constant_feature_maps_to_zero_not_nan() {
        let data = [5.0_f64, 1.0, 5.0, 2.0];
        let out = StandardScaler::new()
            .fit_transform(view(&data, 2, 2))
            .unwrap();
        assert_eq!(out.get(0, 0), 0.0);
        assert!(out.into_vec().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn minmax_maps_to_unit_interval_and_inverts() {
        let data = [2.0_f64, -4.0, 4.0, 0.0, 6.0, 4.0];
        let mut s = MinMaxScaler::new();
        let out = s.fit_transform(view(&data, 3, 2)).unwrap();
        assert_eq!(out.get(0, 0), 0.0);
        assert_eq!(out.get(2, 0), 1.0);
        let back = s.inverse_transform(&out).unwrap();
        for (a, b) in back.into_vec().iter().zip(&data) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn robust_scaler_centres_on_median_and_ignores_outlier_scale() {
        // Sorted: 1..5, 1000. Linear quantiles: median 3.5, q25 2.25, q75 4.75, IQR 2.5.
        let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 1000.0];
        let out = RobustScaler::new()
            .fit_transform(view(&data, 6, 1))
            .unwrap();
        assert!((out.get(2, 0) - (3.0 - 3.5) / 2.5).abs() < 1e-12);
        // The outlier stays far out, but the scale was not dragged by it.
        assert!((out.get(5, 0) - (1000.0 - 3.5) / 2.5).abs() < 1e-9);
    }

    #[test]
    fn transform_before_fit_and_width_mismatch_are_errors() {
        let data = [1.0_f64, 2.0, 3.0, 4.0];
        let s = StandardScaler::new();
        assert_eq!(
            s.transform(view(&data, 2, 2)).unwrap_err(),
            Error::NotFitted
        );
        let mut s = StandardScaler::new();
        UnsupervisedEstimator::fit(&mut s, view(&data, 2, 2)).unwrap();
        assert!(matches!(
            s.transform(view(&data, 4, 1)).unwrap_err(),
            Error::Shape(_)
        ));
    }
}

#[cfg(test)]
mod persist_tests {
    use super::*;
    use tpt_ml_proven_core::{DenseView, Transformer, UnsupervisedEstimator};

    fn data() -> Vec<f64> {
        vec![1.0, 10.0, 2.0, 30.0, 4.0, 10.0, 7.0, 50.0, 3.0, 20.0]
    }

    fn same<T: Transformer<Float = f64, Output = Dense<f64>>>(a: &T, b: &T) -> bool {
        let x = data();
        let u = a.transform(DenseView::new(&x, 5, 2).unwrap()).unwrap();
        let v = b.transform(DenseView::new(&x, 5, 2).unwrap()).unwrap();
        u == v
    }

    #[test]
    fn standard_scaler_round_trips() {
        let x = data();
        let mut s = StandardScaler::<f64>::new();
        UnsupervisedEstimator::fit(&mut s, DenseView::new(&x, 5, 2).unwrap()).unwrap();
        let loaded = StandardScaler::<f64>::from_bytes(&s.to_bytes().unwrap()).unwrap();
        assert_eq!(loaded, s);
        assert!(same(&s, &loaded));
    }

    #[test]
    fn minmax_and_robust_scalers_round_trip() {
        let x = data();
        let mut m = MinMaxScaler::<f64>::new();
        UnsupervisedEstimator::fit(&mut m, DenseView::new(&x, 5, 2).unwrap()).unwrap();
        let loaded = MinMaxScaler::<f64>::from_bytes(&m.to_bytes().unwrap()).unwrap();
        assert_eq!(loaded, m);
        assert!(same(&m, &loaded));

        let mut r = RobustScaler::<f64>::new();
        UnsupervisedEstimator::fit(&mut r, DenseView::new(&x, 5, 2).unwrap()).unwrap();
        let loaded = RobustScaler::<f64>::from_bytes(&r.to_bytes().unwrap()).unwrap();
        assert_eq!(loaded, r);
        assert!(same(&r, &loaded));
    }

    #[test]
    fn unfitted_scalers_round_trip() {
        let s = StandardScaler::<f64>::new();
        assert_eq!(
            StandardScaler::<f64>::from_bytes(&s.to_bytes().unwrap()).unwrap(),
            s
        );
    }
}
