// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! One dependency for the whole library.
//!
//! Core traits, metrics, model selection and pipelines are always available. Each estimator
//! family sits behind a feature of the same name, and all are on by default. Turn off
//! `default-features` to keep only the families you use.
//!
//! ```
//! use tpt_ml_proven::prelude::*;
//!
//! let x = [1.0_f64, 10.0, 3.0, 30.0];
//! let y = [2.0_f64, 4.0];
//! let mut model = StandardScaler::<f64>::new().then(Ridge::<f64>::default());
//! model.fit(DenseView::new(&x, 2, 2).unwrap(), &y[..]).unwrap();
//! let r2 = model.predict(DenseView::new(&x, 2, 2).unwrap()).unwrap();
//! assert_eq!(r2.len(), 2);
//! ```

pub use tpt_ml_proven_core::metrics;
pub use tpt_ml_proven_core::model_selection;
pub use tpt_ml_proven_core::persist;
pub use tpt_ml_proven_core::{
    ArrayView, Dense, DenseView, Error, Float, Pipeline, Predictor, ShapeError,
    SupervisedEstimator, TargetView, Then, Transformer, UnsupervisedEstimator,
};

#[cfg(feature = "clustering")]
pub use tpt_ml_proven_clustering as clustering;
#[cfg(feature = "decomposition")]
pub use tpt_ml_proven_decomposition as decomposition;
#[cfg(feature = "ensemble")]
pub use tpt_ml_proven_ensemble as ensemble;
#[cfg(feature = "linear")]
pub use tpt_ml_proven_linear as linear;
#[cfg(feature = "preprocessing")]
pub use tpt_ml_proven_preprocessing as preprocessing;
#[cfg(feature = "trees")]
pub use tpt_ml_proven_trees as trees;

/// The types most programs need, in one import: `use tpt_ml_proven::prelude::*;`.
pub mod prelude {
    pub use crate::{
        ArrayView, Dense, DenseView, Error, Float, Pipeline, Predictor, SupervisedEstimator,
        TargetView, Then, Transformer, UnsupervisedEstimator,
    };

    #[cfg(feature = "clustering")]
    pub use crate::clustering::{Dbscan, Hdbscan, KMeans};
    #[cfg(feature = "decomposition")]
    pub use crate::decomposition::{Nmf, Pca, TruncatedSvd};
    #[cfg(feature = "ensemble")]
    pub use crate::ensemble::StumpForest;
    #[cfg(feature = "linear")]
    pub use crate::linear::{ElasticNet, Lasso, LogisticRegression, MulticlassLogistic, Ridge};
    #[cfg(feature = "preprocessing")]
    pub use crate::preprocessing::{
        ImputeStrategy, MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, SimpleImputer,
        StandardScaler,
    };
    #[cfg(feature = "trees")]
    pub use crate::trees::{DecisionTreeRegressor, GradientBoostingRegressor};
}
