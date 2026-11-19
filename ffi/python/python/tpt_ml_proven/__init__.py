"""Classical machine learning with contract-checked algorithms.

The estimators are implemented in Rust (in the compiled `_native` module) and re-exported
here. The package ships type stubs (`_native.pyi`) and a `py.typed` marker.
"""

from ._native import (
    DBSCAN,
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
    "HDBSCAN",
    "NMF",
    "PCA",
    "KMeans",
    "ElasticNet",
    "Lasso",
    "LogisticRegression",
    "Ridge",
    "StumpForest",
    "TruncatedSVD",
]
