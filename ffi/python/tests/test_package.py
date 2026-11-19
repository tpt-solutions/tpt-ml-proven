"""The package layout: the re-exporting package, the compiled module, the stubs and the marker."""

from importlib import resources

import tpt_ml_proven
from tpt_ml_proven import _native

EXPORTED = [
    "DBSCAN",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "OneHotEncoder",
    "OrdinalEncoder",
    "SimpleImputer",
    "DecisionTreeRegressor",
    "DecisionTreeClassifier",
    "GradientBoostingRegressor",
    "RandomForestRegressor",
    "RandomForestClassifier",

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


def test_every_estimator_is_re_exported_from_the_native_module():
    for name in EXPORTED:
        assert getattr(tpt_ml_proven, name) is getattr(_native, name), name
    assert sorted(tpt_ml_proven.__all__) == sorted(EXPORTED)


def test_the_package_is_marked_typed_and_ships_stubs():
    files = resources.files("tpt_ml_proven")
    assert (files / "py.typed").is_file()
    stub = (files / "_native.pyi").read_text(encoding="utf-8")
    for name in EXPORTED:
        assert f"class {name}" in stub, name
