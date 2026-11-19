// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Model files for DBSCAN and HDBSCAN, in the format described by `tpt_ml_proven_core::persist`.
//! The settings are stored with the fitted labels, so a loaded model reports its settings.

use tpt_ml_proven_core::error::Error;
use tpt_ml_proven_core::persist::{kind, Reader, Writer};
use tpt_ml_proven_core::Float;

use crate::dbscan::Dbscan;
use crate::hdbscan::Hdbscan;

fn invalid(reason: &'static str) -> Error {
    Error::Format { reason }
}

/// Labels are `-1` for noise or a cluster index below the cluster count.
fn check_labels(labels: &[i64]) -> Result<(), Error> {
    if labels.iter().any(|&l| l < -1) {
        return Err(invalid("cluster label is below -1"));
    }
    Ok(())
}

impl<F: Float> Dbscan<F> {
    /// Serialises the model (settings and fitted labels) to a model file.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut w = Writer::for_float::<F>(kind::DBSCAN);
        w.f64(self.eps.to_f64());
        w.usize(self.min_samples);
        match (&self.labels, &self.core) {
            (Some(labels), Some(core)) => {
                w.bool(true);
                w.usize(labels.len());
                for &l in labels {
                    w.f64(l as f64);
                }
                w.usizes(core);
            }
            _ => w.bool(false),
        }
        Ok(w.finish())
    }

    /// Loads a model written by [`Dbscan::to_bytes`], checking the settings and that core
    /// indices point at labelled rows.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(bytes, kind::DBSCAN)?;
        let eps = r.f64()?;
        let min_samples = r.usize()?;
        let fitted = if r.bool()? {
            let n = r.usize()?;
            let mut labels = Vec::new();
            for _ in 0..n {
                let v = r.f64()?;
                if v.fract() != 0.0 {
                    return Err(invalid("DBSCAN label is not a whole number"));
                }
                labels.push(v as i64);
            }
            let core = r.usizes()?;
            Some((labels, core))
        } else {
            None
        };
        r.finish()?;
        if eps.is_nan() || eps <= 0.0 || min_samples == 0 {
            return Err(invalid("DBSCAN settings are out of range"));
        }
        let mut model = Dbscan::<F>::new(F::from_f64(eps), min_samples);
        if let Some((labels, core)) = fitted {
            check_labels(&labels)?;
            if core.iter().any(|&i| i >= labels.len()) || core.windows(2).any(|w| w[0] >= w[1]) {
                return Err(invalid(
                    "DBSCAN core indices are not sorted rows of the fit",
                ));
            }
            model.labels = Some(labels);
            model.core = Some(core);
        }
        Ok(model)
    }
}

impl<F: Float> Hdbscan<F> {
    /// Serialises the model (settings and fitted labels) to a model file.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut w = Writer::for_float::<F>(kind::HDBSCAN);
        w.usize(self.min_cluster_size);
        match self.min_samples {
            None => w.bool(false),
            Some(m) => {
                w.bool(true);
                w.usize(m);
            }
        }
        match &self.labels {
            None => w.bool(false),
            Some(labels) => {
                w.bool(true);
                w.usize(labels.len());
                for &l in labels {
                    w.f64(l as f64);
                }
            }
        }
        Ok(w.finish())
    }

    /// Loads a model written by [`Hdbscan::to_bytes`], checking the settings and the labels.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(bytes, kind::HDBSCAN)?;
        let min_cluster_size = r.usize()?;
        let min_samples = if r.bool()? { Some(r.usize()?) } else { None };
        let labels = if r.bool()? {
            let n = r.usize()?;
            let mut labels = Vec::new();
            for _ in 0..n {
                let v = r.f64()?;
                if v.fract() != 0.0 {
                    return Err(invalid("HDBSCAN label is not a whole number"));
                }
                labels.push(v as i64);
            }
            Some(labels)
        } else {
            None
        };
        r.finish()?;
        if min_cluster_size < 2 {
            return Err(invalid("HDBSCAN min_cluster_size must be at least 2"));
        }
        if min_samples == Some(0) {
            return Err(invalid("HDBSCAN min_samples must be at least 1"));
        }
        let mut model = Hdbscan::<F>::new(min_cluster_size);
        if let Some(m) = min_samples {
            model = model.with_min_samples(m);
        }
        if let Some(labels) = labels {
            check_labels(&labels)?;
            model.labels = Some(labels);
        }
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dbscan_round_trips_with_its_labels() {
        let x = [0.0_f64, 0.0, 0.1, 0.0, 5.0, 5.0, 5.1, 5.0];
        let view = tpt_ml_proven_core::DenseView::new(&x, 4, 2).unwrap();
        let mut m = Dbscan::<f64>::new(0.5, 2);
        tpt_ml_proven_core::UnsupervisedEstimator::fit(&mut m, view).unwrap();
        let loaded = Dbscan::<f64>::from_bytes(&m.to_bytes().unwrap()).unwrap();
        assert_eq!(loaded.labels(), m.labels());
        assert_eq!(loaded.core_sample_indices(), m.core_sample_indices());
        assert_eq!(loaded.to_bytes().unwrap(), m.to_bytes().unwrap());
    }

    #[test]
    fn unfitted_hdbscan_round_trips_its_settings() {
        let m = Hdbscan::<f64>::new(4).with_min_samples(3);
        let loaded = Hdbscan::<f64>::from_bytes(&m.to_bytes().unwrap()).unwrap();
        assert_eq!(loaded.to_bytes().unwrap(), m.to_bytes().unwrap());
    }

    #[test]
    fn a_core_index_outside_the_labels_is_refused() {
        let mut w = Writer::for_float::<f64>(kind::DBSCAN);
        w.f64(0.5);
        w.usize(2);
        w.bool(true);
        w.usize(1);
        w.f64(0.0);
        w.usizes(&[3]);
        assert!(Dbscan::<f64>::from_bytes(&w.finish()).is_err());
    }

    #[test]
    fn a_label_below_noise_is_refused() {
        let mut w = Writer::for_float::<f64>(kind::HDBSCAN);
        w.usize(3);
        w.bool(false);
        w.bool(true);
        w.usize(1);
        w.f64(-2.0);
        assert!(Hdbscan::<f64>::from_bytes(&w.finish()).is_err());
    }
}
