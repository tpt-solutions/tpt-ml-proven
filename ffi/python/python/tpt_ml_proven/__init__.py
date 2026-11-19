"""Classical machine learning with contract-checked algorithms.

The estimators are implemented in Rust (in the compiled `_native` module) and re-exported
here. The package ships type stubs (`_native.pyi`) and a `py.typed` marker.
"""

from ._native import (
    DBSCAN,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
    SimpleImputer,
    DecisionTreeRegressor,
    DecisionTreeClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
    RandomForestClassifier,

    HDBSCAN,
    NMF,
    PCA,
    KMeans,
    ElasticNet,
    Lasso,
    LogisticRegression,
    Ridge,
    StumpForest,
    TruncatedSVD,
)

__all__ = [
    "DBSCAN",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ElasticNet",
    "GradientBoostingRegressor",
    "HDBSCAN",
    "KMeans",
    "Lasso",
    "LogisticRegression",
    "MinMaxScaler",
    "NMF",
    "OneHotEncoder",
    "OrdinalEncoder",
    "PCA",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "Ridge",
    "RobustScaler",
    "SimpleImputer",
    "StandardScaler",
    "StumpForest",
    "TruncatedSVD",
]
