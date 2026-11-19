// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Preprocessing in WebAssembly: the scalers, the encoders and the imputer.
//!
//! Every transformer has `fit(data, cols)`, `transform(data, cols)` and `fitTransform(data,
//! cols)`. The output is flat, with `outputCols` values per row. For the scalers and the
//! imputer that is the input width, and for the one-hot encoder it is the number of indicators.

use wasm_bindgen::prelude::*;

use tpt_ml_proven_core::{Transformer, UnsupervisedEstimator};
use tpt_ml_proven_preprocessing::{
    ImputeStrategy, MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, SimpleImputer,
    StandardScaler,
};

use crate::options::Options;
use crate::{flatten, js_err, not_fitted, view};

/// Defines a transformer with no options: `fit`, `transform`, `fitTransform` and `outputCols`.
/// `$width` gives the output width of a fitted model.
macro_rules! transformer_class {
    ($name:ident, $rust:ident, $doc:literal, |$m:ident| $width:expr) => {
        #[doc = $doc]
        #[wasm_bindgen]
        pub struct $name {
            fitted: Option<$rust<f64>>,
        }

        #[wasm_bindgen]
        impl $name {
            #[wasm_bindgen(constructor)]
            pub fn new(options: Option<JsValue>) -> Result<$name, JsError> {
                Options::new(options)?;
                Ok($name { fitted: None })
            }

            /// Serialises the fitted transformer as model-file bytes.
            #[wasm_bindgen(js_name = toBytes)]
            pub fn to_bytes(&self) -> Result<Vec<u8>, JsError> {
                self.fitted
                    .as_ref()
                    .ok_or_else(not_fitted)?
                    .to_bytes()
                    .map_err(js_err)
            }

            /// Loads a model written by `toBytes`.
            #[wasm_bindgen(js_name = fromBytes)]
            pub fn from_bytes(bytes: &[u8]) -> Result<$name, JsError> {
                Ok($name {
                    fitted: Some($rust::<f64>::from_bytes(bytes).map_err(js_err)?),
                })
            }

            /// Learns the transformation from a flat row-major matrix.
            pub fn fit(&mut self, data: &[f64], cols: usize) -> Result<(), JsError> {
                let mut m = $rust::<f64>::new();
                UnsupervisedEstimator::fit(&mut m, view(data, cols)?).map_err(js_err)?;
                self.fitted = Some(m);
                Ok(())
            }

            /// Applies the fitted transformation, flat `rows × outputCols`.
            pub fn transform(&self, data: &[f64], cols: usize) -> Result<Vec<f64>, JsError> {
                let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
                Ok(flatten(
                    &Transformer::transform(m, view(data, cols)?).map_err(js_err)?,
                ))
            }

            /// Fits, then transforms the same data.
            #[wasm_bindgen(js_name = fitTransform)]
            pub fn fit_transform(
                &mut self,
                data: &[f64],
                cols: usize,
            ) -> Result<Vec<f64>, JsError> {
                self.fit(data, cols)?;
                self.transform(data, cols)
            }

            /// Values per output row, once fitted.
            #[wasm_bindgen(getter, js_name = outputCols)]
            pub fn output_cols(&self) -> Result<usize, JsError> {
                let $m = self.fitted.as_ref().ok_or_else(not_fitted)?;
                Ok($width)
            }
        }
    };
}

transformer_class!(
    StandardScalerModel,
    StandardScaler,
    "Standardises each feature to mean 0 and unit variance (constant features map to 0).",
    |m| m.n_features().unwrap_or(0)
);

transformer_class!(
    MinMaxScalerModel,
    MinMaxScaler,
    "Rescales each feature to `[0, 1]`.",
    |m| m.n_features().unwrap_or(0)
);

transformer_class!(
    RobustScalerModel,
    RobustScaler,
    "Centres each feature on its median and scales by the interquartile range.",
    |m| m.n_features().unwrap_or(0)
);

transformer_class!(
    OneHotEncoderModel,
    OneHotEncoder,
    "Encodes each category as an indicator column. Unknown categories are an error.",
    |m| m.categories().map_or(0, |c| c.iter().map(Vec::len).sum())
);

transformer_class!(
    OrdinalEncoderModel,
    OrdinalEncoder,
    "Encodes each category as its index in the sorted list of training categories.",
    |m| m.categories().map_or(0, |c| c.len())
);

/// Fills missing (NaN) entries with a per-column statistic learned in `fit`.
///
/// Options: `strategy` (`"mean"`, `"median"`, `"most_frequent"` or `"constant"`; default
/// `"mean"`), and `fillValue` (0), used by the `constant` strategy.
#[wasm_bindgen]
pub struct SimpleImputerModel {
    strategy: ImputeStrategy,
    fitted: Option<SimpleImputer<f64>>,
}

#[wasm_bindgen]
impl SimpleImputerModel {
    #[wasm_bindgen(constructor)]
    pub fn new(options: Option<JsValue>) -> Result<SimpleImputerModel, JsError> {
        let o = Options::new(options)?;
        let fill = o.f64_or("fillValue", 0.0)?;
        let strategy = match o.str_or("strategy", "mean")?.as_str() {
            "mean" => ImputeStrategy::Mean,
            "median" => ImputeStrategy::Median,
            "most_frequent" => ImputeStrategy::MostFrequent,
            "constant" => ImputeStrategy::Constant(fill),
            other => {
                return Err(JsError::new(&format!(
                "strategy must be 'mean', 'median', 'most_frequent' or 'constant', not {other:?}"
            )))
            }
        };
        Ok(SimpleImputerModel {
            strategy,
            fitted: None,
        })
    }

    /// Learns the fill values from a flat row-major matrix; NaN entries are ignored.
    pub fn fit(&mut self, data: &[f64], cols: usize) -> Result<(), JsError> {
        let mut m = SimpleImputer::<f64>::new(self.strategy);
        UnsupervisedEstimator::fit(&mut m, view(data, cols)?).map_err(js_err)?;
        self.fitted = Some(m);
        Ok(())
    }

    /// The matrix with NaN entries replaced by the fill values, flat `rows × cols`.
    pub fn transform(&self, data: &[f64], cols: usize) -> Result<Vec<f64>, JsError> {
        let m = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(flatten(
            &Transformer::transform(m, view(data, cols)?).map_err(js_err)?,
        ))
    }

    /// Fits, then transforms the same data.
    #[wasm_bindgen(js_name = fitTransform)]
    pub fn fit_transform(&mut self, data: &[f64], cols: usize) -> Result<Vec<f64>, JsError> {
        self.fit(data, cols)?;
        self.transform(data, cols)
    }

    /// Values per output row, once fitted: the input width.
    #[wasm_bindgen(getter, js_name = outputCols)]
    pub fn output_cols(&self) -> Result<usize, JsError> {
        Ok(self
            .fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .statistics()
            .map_or(0, <[f64]>::len))
    }
}

#[wasm_bindgen]
impl SimpleImputerModel {
    /// Serialises the fitted imputer as model-file bytes.
    #[wasm_bindgen(js_name = toBytes)]
    pub fn to_bytes(&self) -> Result<Vec<u8>, JsError> {
        self.fitted
            .as_ref()
            .ok_or_else(not_fitted)?
            .to_bytes()
            .map_err(js_err)
    }

    /// Loads a model written by `toBytes`. The strategy is read from the file.
    #[wasm_bindgen(js_name = fromBytes)]
    pub fn from_bytes(bytes: &[u8]) -> Result<SimpleImputerModel, JsError> {
        let m = SimpleImputer::<f64>::from_bytes(bytes).map_err(js_err)?;
        Ok(SimpleImputerModel {
            strategy: m.strategy(),
            fitted: Some(m),
        })
    }
}
