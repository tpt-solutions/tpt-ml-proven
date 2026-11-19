// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! The models the command line can load, fit, predict with and transform with. A model file
//! names its own kind in the header, so [`Model::from_bytes`] picks the right loader.

use tpt_ml_proven::persist::{kind, MAGIC};
use tpt_ml_proven::prelude::*;

use crate::args::Args;
use crate::csv::Table;

/// Any model the command line knows, all over `f64`.
pub enum Model {
    /// A fitted KMeans model.
    KMeans(KMeans<f64>),
    /// A fitted Pca model.
    Pca(Pca<f64>),
    /// A fitted TruncatedSvd model.
    TruncatedSvd(TruncatedSvd<f64>),
    /// A fitted Nmf model.
    Nmf(Nmf<f64>),
    /// A fitted Ridge model.
    Ridge(Ridge<f64>),
    /// A fitted Lasso model.
    Lasso(Lasso<f64>),
    /// A fitted ElasticNet model.
    ElasticNet(ElasticNet<f64>),
    /// A fitted Logistic model.
    Logistic(LogisticRegression<f64>),
    /// A fitted Multiclass model.
    Multiclass(MulticlassLogistic<f64>),
    /// A fitted StumpForest model.
    StumpForest(StumpForest<f64>),
    /// A fitted DecisionTree model.
    DecisionTree(DecisionTreeRegressor),
    /// A fitted GradientBoosting model.
    GradientBoosting(GradientBoostingRegressor),
    /// A fitted StandardScaler model.
    StandardScaler(StandardScaler<f64>),
    /// A fitted MinMaxScaler model.
    MinMaxScaler(MinMaxScaler<f64>),
    /// A fitted RobustScaler model.
    RobustScaler(RobustScaler<f64>),
    /// A fitted OneHotEncoder model.
    OneHotEncoder(OneHotEncoder<f64>),
    /// A fitted OrdinalEncoder model.
    OrdinalEncoder(OrdinalEncoder<f64>),
    /// A fitted SimpleImputer model.
    SimpleImputer(SimpleImputer<f64>),
}

/// Human-readable name of a model kind tag, or `None` for an unknown tag.
pub fn kind_name(tag: u32) -> Option<&'static str> {
    Some(match tag {
        kind::KMEANS => "kmeans",
        kind::PCA => "pca",
        kind::RIDGE => "ridge",
        kind::LASSO => "lasso",
        kind::ELASTIC_NET => "elasticnet",
        kind::LOGISTIC => "logistic",
        kind::MULTICLASS_LOGISTIC => "multiclass-logistic",
        kind::DECISION_TREE => "decision-tree",
        kind::GRADIENT_BOOSTING => "gradient-boosting",
        kind::TRUNCATED_SVD => "truncated-svd",
        kind::NMF => "nmf",
        kind::STUMP_FOREST => "stump-forest",
        kind::STANDARD_SCALER => "standard-scaler",
        kind::MINMAX_SCALER => "minmax-scaler",
        kind::ROBUST_SCALER => "robust-scaler",
        kind::ONE_HOT_ENCODER => "one-hot-encoder",
        kind::ORDINAL_ENCODER => "ordinal-encoder",
        kind::SIMPLE_IMPUTER => "simple-imputer",
        _ => return None,
    })
}

/// The header's kind tag, after checking the magic bytes. Full validation happens in the loader.
pub fn file_kind(bytes: &[u8]) -> Result<u32, String> {
    if bytes.len() < 16 {
        return Err("file is too short to be a model".to_string());
    }
    if bytes[..8] != MAGIC {
        return Err("not a tpt-ml-proven model file".to_string());
    }
    Ok(u32::from_le_bytes(
        bytes[12..16].try_into().expect("4 bytes"),
    ))
}

fn text<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

impl Model {
    /// The kind name, as `info` prints it.
    pub fn name(&self) -> &'static str {
        match self {
            Model::KMeans(_) => "kmeans",
            Model::Pca(_) => "pca",
            Model::TruncatedSvd(_) => "truncated-svd",
            Model::Nmf(_) => "nmf",
            Model::Ridge(_) => "ridge",
            Model::Lasso(_) => "lasso",
            Model::ElasticNet(_) => "elasticnet",
            Model::Logistic(_) => "logistic",
            Model::Multiclass(_) => "multiclass-logistic",
            Model::StumpForest(_) => "stump-forest",
            Model::DecisionTree(_) => "decision-tree",
            Model::GradientBoosting(_) => "gradient-boosting",
            Model::StandardScaler(_) => "standard-scaler",
            Model::MinMaxScaler(_) => "minmax-scaler",
            Model::RobustScaler(_) => "robust-scaler",
            Model::OneHotEncoder(_) => "one-hot-encoder",
            Model::OrdinalEncoder(_) => "ordinal-encoder",
            Model::SimpleImputer(_) => "simple-imputer",
        }
    }

    /// Loads any model file, choosing the loader from its kind tag.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let tag = file_kind(bytes)?;
        Ok(match tag {
            kind::KMEANS => Model::KMeans(KMeans::from_bytes(bytes).map_err(text)?),
            kind::PCA => Model::Pca(Pca::from_bytes(bytes).map_err(text)?),
            kind::TRUNCATED_SVD => {
                Model::TruncatedSvd(TruncatedSvd::from_bytes(bytes).map_err(text)?)
            }
            kind::NMF => Model::Nmf(Nmf::from_bytes(bytes).map_err(text)?),
            kind::RIDGE => Model::Ridge(Ridge::from_bytes(bytes).map_err(text)?),
            kind::LASSO => Model::Lasso(Lasso::from_bytes(bytes).map_err(text)?),
            kind::ELASTIC_NET => Model::ElasticNet(ElasticNet::from_bytes(bytes).map_err(text)?),
            kind::LOGISTIC => Model::Logistic(LogisticRegression::from_bytes(bytes).map_err(text)?),
            kind::MULTICLASS_LOGISTIC => {
                Model::Multiclass(MulticlassLogistic::from_bytes(bytes).map_err(text)?)
            }
            kind::STUMP_FOREST => Model::StumpForest(StumpForest::from_bytes(bytes).map_err(text)?),
            kind::DECISION_TREE => {
                Model::DecisionTree(DecisionTreeRegressor::from_bytes(bytes).map_err(text)?)
            }
            kind::GRADIENT_BOOSTING => {
                Model::GradientBoosting(GradientBoostingRegressor::from_bytes(bytes).map_err(text)?)
            }
            kind::STANDARD_SCALER => {
                Model::StandardScaler(StandardScaler::from_bytes(bytes).map_err(text)?)
            }
            kind::MINMAX_SCALER => {
                Model::MinMaxScaler(MinMaxScaler::from_bytes(bytes).map_err(text)?)
            }
            kind::ROBUST_SCALER => {
                Model::RobustScaler(RobustScaler::from_bytes(bytes).map_err(text)?)
            }
            kind::ONE_HOT_ENCODER => {
                Model::OneHotEncoder(OneHotEncoder::from_bytes(bytes).map_err(text)?)
            }
            kind::ORDINAL_ENCODER => {
                Model::OrdinalEncoder(OrdinalEncoder::from_bytes(bytes).map_err(text)?)
            }
            kind::SIMPLE_IMPUTER => {
                Model::SimpleImputer(SimpleImputer::from_bytes(bytes).map_err(text)?)
            }
            other => return Err(format!("unknown model kind {other}")),
        })
    }

    /// Serialises the model to a model file.
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        match self {
            Model::KMeans(m) => m.to_bytes(),
            Model::Pca(m) => m.to_bytes(),
            Model::TruncatedSvd(m) => m.to_bytes(),
            Model::Nmf(m) => m.to_bytes(),
            Model::Ridge(m) => m.to_bytes(),
            Model::Lasso(m) => m.to_bytes(),
            Model::ElasticNet(m) => m.to_bytes(),
            Model::Logistic(m) => m.to_bytes(),
            Model::Multiclass(m) => m.to_bytes(),
            Model::StumpForest(m) => m.to_bytes(),
            Model::DecisionTree(m) => m.to_bytes(),
            Model::GradientBoosting(m) => m.to_bytes(),
            Model::StandardScaler(m) => m.to_bytes(),
            Model::MinMaxScaler(m) => m.to_bytes(),
            Model::RobustScaler(m) => m.to_bytes(),
            Model::OneHotEncoder(m) => m.to_bytes(),
            Model::OrdinalEncoder(m) => m.to_bytes(),
            Model::SimpleImputer(m) => m.to_bytes(),
        }
        .map_err(text)
    }

    /// Fits a new model. `estimator` is one of the names in [`ESTIMATORS`]; supervised ones
    /// need `y`. Hyper-parameters come from the flags, with the defaults shown in `--help`.
    pub fn fit(estimator: &str, x: &Table, y: Option<&[f64]>, opts: &Args) -> Result<Self, String> {
        let view = || DenseView::new(&x.data, x.rows, x.cols).map_err(text);
        let target = || -> Result<&[f64], String> {
            let y = y.ok_or_else(|| format!("{estimator} needs a target: pass --target FILE"))?;
            if y.len() != x.rows {
                return Err(format!(
                    "target has {} values but the input has {} rows",
                    y.len(),
                    x.rows
                ));
            }
            Ok(y)
        };
        let unsupervised = |fit: Result<(), Error>| fit.map_err(text);
        Ok(match estimator {
            "kmeans" => {
                let mut m = KMeans::<f64>::new(opts.parsed_or("k", 8)?)
                    .with_seed(opts.parsed_or("seed", 0)?);
                unsupervised(UnsupervisedEstimator::fit(&mut m, view()?))?;
                Model::KMeans(m)
            }
            "pca" => {
                let mut m = Pca::<f64>::new(opts.parsed_or("components", 2)?);
                unsupervised(UnsupervisedEstimator::fit(&mut m, view()?))?;
                Model::Pca(m)
            }
            "svd" => {
                let mut m = TruncatedSvd::<f64>::new(opts.parsed_or("components", 2)?);
                unsupervised(UnsupervisedEstimator::fit(&mut m, view()?))?;
                Model::TruncatedSvd(m)
            }
            "nmf" => {
                let mut m = Nmf::<f64>::new(opts.parsed_or("components", 2)?)
                    .with_seed(opts.parsed_or("seed", 0)?);
                unsupervised(UnsupervisedEstimator::fit(&mut m, view()?))?;
                Model::Nmf(m)
            }
            "standard-scaler" => {
                let mut m = StandardScaler::<f64>::new();
                unsupervised(UnsupervisedEstimator::fit(&mut m, view()?))?;
                Model::StandardScaler(m)
            }
            "minmax-scaler" => {
                let mut m = MinMaxScaler::<f64>::new();
                unsupervised(UnsupervisedEstimator::fit(&mut m, view()?))?;
                Model::MinMaxScaler(m)
            }
            "ridge" => {
                let mut m = Ridge::<f64>::new(opts.parsed_or("alpha", 1.0)?);
                SupervisedEstimator::fit(&mut m, view()?, target()?).map_err(text)?;
                Model::Ridge(m)
            }
            "lasso" => {
                let mut m = Lasso::<f64>::new(opts.parsed_or("alpha", 1.0)?);
                SupervisedEstimator::fit(&mut m, view()?, target()?).map_err(text)?;
                Model::Lasso(m)
            }
            "elasticnet" => {
                let mut m = ElasticNet::<f64>::new(
                    opts.parsed_or("alpha", 1.0)?,
                    opts.parsed_or("l1-ratio", 0.5)?,
                );
                SupervisedEstimator::fit(&mut m, view()?, target()?).map_err(text)?;
                Model::ElasticNet(m)
            }
            "logistic" => {
                let mut m = LogisticRegression::<f64>::new(opts.parsed_or("c", 1.0)?);
                SupervisedEstimator::fit(&mut m, view()?, target()?).map_err(text)?;
                Model::Logistic(m)
            }
            other => return Err(format!("unknown estimator '{other}'")),
        })
    }

    /// Predictions, one value per row. Labels are returned as `f64`.
    pub fn predict(&self, x: &Table) -> Result<Vec<f64>, String> {
        let view = DenseView::new(&x.data, x.rows, x.cols).map_err(text)?;
        match self {
            Model::KMeans(m) => Predictor::predict(m, view)
                .map(|v| v.into_iter().map(|l| l as f64).collect())
                .map_err(text),
            Model::Ridge(m) => Predictor::predict(m, view).map_err(text),
            Model::Lasso(m) => Predictor::predict(m, view).map_err(text),
            Model::ElasticNet(m) => Predictor::predict(m, view).map_err(text),
            Model::Logistic(m) => Predictor::predict(m, view).map_err(text),
            Model::Multiclass(m) => Predictor::predict(m, view).map_err(text),
            Model::StumpForest(m) => Predictor::predict(m, view).map_err(text),
            Model::DecisionTree(m) => Predictor::predict(m, view).map_err(text),
            Model::GradientBoosting(m) => Predictor::predict(m, view).map_err(text),
            other => Err(format!("{} does not predict; use transform", other.name())),
        }
    }

    /// Transformed rows, as a matrix.
    pub fn transform(&self, x: &Table) -> Result<Dense<f64>, String> {
        let view = DenseView::new(&x.data, x.rows, x.cols).map_err(text)?;
        match self {
            Model::Pca(m) => Transformer::transform(m, view).map_err(text),
            Model::TruncatedSvd(m) => Transformer::transform(m, view).map_err(text),
            Model::Nmf(m) => Transformer::transform(m, view).map_err(text),
            Model::StandardScaler(m) => Transformer::transform(m, view).map_err(text),
            Model::MinMaxScaler(m) => Transformer::transform(m, view).map_err(text),
            Model::RobustScaler(m) => Transformer::transform(m, view).map_err(text),
            Model::OneHotEncoder(m) => Transformer::transform(m, view).map_err(text),
            Model::OrdinalEncoder(m) => Transformer::transform(m, view).map_err(text),
            Model::SimpleImputer(m) => Transformer::transform(m, view).map_err(text),
            other => Err(format!("{} does not transform; use predict", other.name())),
        }
    }

    /// True for models whose output is `predict`, false for those that use `transform`.
    pub fn predicts(&self) -> bool {
        !matches!(
            self,
            Model::Pca(_)
                | Model::TruncatedSvd(_)
                | Model::Nmf(_)
                | Model::StandardScaler(_)
                | Model::MinMaxScaler(_)
                | Model::RobustScaler(_)
                | Model::OneHotEncoder(_)
                | Model::OrdinalEncoder(_)
                | Model::SimpleImputer(_)
        )
    }
}

/// Estimator names accepted by `fit`.
pub const ESTIMATORS: &[&str] = &[
    "kmeans",
    "pca",
    "svd",
    "nmf",
    "standard-scaler",
    "minmax-scaler",
    "ridge",
    "lasso",
    "elasticnet",
    "logistic",
];
