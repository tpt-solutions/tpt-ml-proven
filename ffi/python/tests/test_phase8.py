# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 TPT Solutions
"""Behaviour of the Phase 8 estimator bindings. Run with `pytest` after building the wheel."""

import numpy as np
import pytest

from tpt_ml_proven import (
    DBSCAN,
    HDBSCAN,
    NMF,
    ElasticNet,
    Lasso,
    LogisticRegression,
    Ridge,
    StumpForest,
    TruncatedSVD,
)


def two_blobs():
    rng = np.random.default_rng(0)
    a = 0.2 * rng.standard_normal((30, 2))
    b = np.array([5.0, 5.0]) + 0.2 * rng.standard_normal((30, 2))
    return np.vstack([a, b])


def test_dbscan_finds_two_clusters_and_noise():
    x = np.vstack([two_blobs(), [[20.0, -20.0]]])
    model = DBSCAN(eps=1.0, min_samples=5).fit(x)
    labels = np.array(model.labels_)
    assert set(labels) == {-1, 0, 1}
    assert labels[-1] == -1
    assert len(model.core_sample_indices_) > 0


def test_hdbscan_finds_two_clusters():
    model = HDBSCAN(min_cluster_size=10).fit(two_blobs())
    labels = np.array(model.labels_)
    assert set(labels) - {-1} == {0, 1}
    assert len(set(labels[:30])) == 1 and len(set(labels[30:])) == 1
    assert labels[0] != labels[30]


def test_truncated_svd_matches_numpy_singular_values():
    x = two_blobs() + 1.0
    model = TruncatedSVD(2).fit(x)
    expected = np.linalg.svd(x, compute_uv=False)[:2]
    assert np.allclose(model.singular_values_, expected, rtol=1e-8)
    assert np.asarray(model.transform(x)).shape == (60, 2)


def test_nmf_factors_are_non_negative():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, size=(20, 3)) @ rng.uniform(0, 1, size=(3, 6))
    model = NMF(3, max_iter=500, seed=2).fit(x)
    assert np.all(np.asarray(model.components_) >= 0)
    w = np.asarray(model.transform(x))
    assert w.shape == (20, 3) and np.all(w >= 0)
    assert model.reconstruction_err_ < 0.5 * np.linalg.norm(x)


def test_ridge_recovers_a_linear_relation():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((50, 3))
    y = x @ np.array([1.0, -2.0, 0.5]) + 0.25
    model = Ridge(1e-9).fit(x, y.tolist())
    assert np.allclose(model.coef_, [1.0, -2.0, 0.5], atol=1e-5)
    assert abs(model.intercept_ - 0.25) < 1e-5
    assert np.allclose(model.predict(x), y, atol=1e-5)


def test_lasso_and_elasticnet_shrink_coefficients():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((60, 4))
    y = (x @ np.array([2.0, 0.0, 0.0, -1.0])).tolist()
    lasso = Lasso(0.5, max_iter=10000, tol=1e-10).fit(x, y)
    enet = ElasticNet(0.5, 0.5, max_iter=10000, tol=1e-10).fit(x, y)
    assert abs(lasso.coef_[1]) < 1e-12
    assert abs(enet.coef_[1]) < 1e-12
    assert abs(lasso.coef_[0]) < 2.0


def test_logistic_binary_keeps_the_original_labels():
    x = two_blobs()
    y = [7.0] * 30 + [3.0] * 30
    model = LogisticRegression(C=10.0).fit(x, y)
    assert model.classes_ == [3.0, 7.0]
    assert model.predict(x) == y
    proba = np.asarray(model.predict_proba(x))
    assert proba.shape == (60, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_logistic_multiclass_separates_three_groups():
    rng = np.random.default_rng(5)
    centres = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    x = np.vstack([c + 0.3 * rng.standard_normal((20, 2)) for c in centres])
    y = [float(i) for i in range(3) for _ in range(20)]
    model = LogisticRegression(C=10.0).fit(x, y)
    assert model.classes_ == [0.0, 1.0, 2.0]
    assert model.predict(x) == y
    assert np.allclose(np.asarray(model.predict_proba(x)).sum(axis=1), 1.0)


def test_stump_forest_fits_a_threshold_rule():
    x = np.arange(40, dtype=float).reshape(40, 1)
    y = [0.0 if v < 20 else 1.0 for v in x[:, 0]]
    model = StumpForest(25, seed=7).fit(x, y)
    assert model.predict(x) == y


def test_float32_input_gives_the_same_ridge_fit_as_float64():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((30, 2))
    y = (x @ np.array([1.5, -0.5])).tolist()
    a = Ridge(0.1).fit(x, y)
    b = Ridge(0.1).fit(x.astype(np.float32), y)
    assert np.allclose(a.coef_, b.coef_, atol=1e-5)


def test_errors_are_raised_as_python_exceptions():
    with pytest.raises(RuntimeError):
        Ridge(0.1).predict(np.zeros((2, 2)))
    with pytest.raises(RuntimeError):
        DBSCAN(eps=1.0, min_samples=2).labels_
    with pytest.raises(ValueError):
        Ridge(0.1).fit(np.zeros((4, 2)), [1.0, 2.0])
    with pytest.raises(ValueError):
        DBSCAN(eps=-1.0, min_samples=2).fit(two_blobs())
