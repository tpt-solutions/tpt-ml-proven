"""get_params, set_params and repr: the constructor parameters, read and set by name."""

import numpy as np
import pytest

from tpt_ml_proven import DBSCAN, HDBSCAN, NMF, PCA, KMeans, ElasticNet, Lasso, LogisticRegression, Ridge, StumpForest, TruncatedSVD

X = np.array([[float(i % 5), float((i * 3) % 4), float(i % 3)] for i in range(24)], dtype=np.float64)
Y = (X[:, 0] - X[:, 1]).tolist()


@pytest.mark.parametrize(
    "model, expected",
    [
        (KMeans(3, seed=2), {"n_clusters": 3, "max_iter": 300, "tol": 1e-4, "init": "k-means++", "seed": 2, "threads": 1}),
        (PCA(2), {"n_components": 2}),
        (TruncatedSVD(2), {"n_components": 2}),
        (NMF(2, seed=3), {"n_components": 2, "max_iter": 200, "tol": 1e-4, "seed": 3}),
        (Ridge(0.5), {"alpha": 0.5}),
        (Lasso(0.25, max_iter=50), {"alpha": 0.25, "max_iter": 50, "tol": 1e-4}),
        (ElasticNet(0.3, 0.7), {"alpha": 0.3, "l1_ratio": 0.7, "max_iter": 1000, "tol": 1e-4}),
        (LogisticRegression(2.0), {"C": 2.0, "max_iter": 100, "tol": 1e-4}),
        (StumpForest(9, seed=1), {"n_estimators": 9, "seed": 1}),
        (DBSCAN(0.5, 4), {"eps": 0.5, "min_samples": 4}),
        (HDBSCAN(6), {"min_cluster_size": 6, "min_samples": None}),
    ],
)
def test_get_params_returns_the_constructor_values(model, expected):
    assert dict(model.get_params()) == expected


def test_set_params_changes_the_value_and_returns_the_estimator():
    model = Ridge(1.0)
    assert model.set_params(alpha=3.0) is model
    assert model.get_params()["alpha"] == 3.0


def test_set_params_on_kmeans_init_is_validated():
    model = KMeans(2)
    model.set_params(init="random")
    assert model.get_params()["init"] == "random"
    with pytest.raises(ValueError):
        model.set_params(init="bogus")


def test_an_unknown_parameter_is_refused():
    with pytest.raises(ValueError, match="invalid parameter"):
        Ridge().set_params(gamma=1.0)


def test_a_changed_parameter_takes_effect_on_refit():
    model = Ridge(1e-9).fit(X, Y)
    before = list(model.coef_)
    model.set_params(alpha=1000.0).fit(X, Y)
    assert list(model.coef_) != before
    assert Ridge(1000.0).fit(X, Y).coef_ == model.coef_


def test_repr_prints_the_class_and_its_parameters():
    assert repr(Ridge(0.5)) == "Ridge(alpha=0.5)"
    assert repr(KMeans(3)).startswith("KMeans(n_clusters=3, max_iter=300")
    assert repr(LogisticRegression(2.0)) == "LogisticRegression(C=2.0, max_iter=100, tol=0.0001)"
