# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 TPT Solutions
"""Behaviour of the Python bindings. Run with `pytest` after `maturin develop`."""

import numpy as np
import pytest

from tpt_ml_proven import KMeans, PCA


def blobs(dtype=np.float64):
    x = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
        ],
        dtype=dtype,
    )
    return x


def canonical(labels):
    seen = []
    out = []
    for label in labels:
        if label not in seen:
            seen.append(label)
        out.append(seen.index(label))
    return out


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kmeans_recovers_blobs_for_both_dtypes(dtype):
    km = KMeans(2, seed=1)
    km.fit(blobs(dtype))
    assert canonical(km.labels_) == [0, 0, 0, 1, 1, 1]
    assert km.converged_
    assert km.inertia_ < 0.1


def test_kmeans_predict_and_transform_shapes():
    x = blobs()
    km = KMeans(2, seed=1)
    km.fit(x)
    assert km.predict(x) == km.labels_
    d = km.transform(x)
    assert len(d) == 6 and len(d[0]) == 2
    assert all(v >= 0 for row in d for v in row)


def test_kmeans_fit_transform_matches_fit_then_transform():
    x = blobs()
    a = KMeans(2, seed=3).fit_transform(x)
    b = KMeans(2, seed=3)
    b.fit(x)
    assert a == b.transform(x)


def test_kmeans_random_init_and_threads_give_same_labels():
    x = np.random.default_rng(0).normal(size=(300, 4))
    one = KMeans(4, init="random", seed=9, threads=1)
    one.fit(x)
    many = KMeans(4, init="random", seed=9, threads=4)
    many.fit(x)
    assert one.labels_ == many.labels_


def test_kmeans_rejects_non_contiguous_and_wrong_rank():
    x = np.zeros((6, 4))[:, ::2]  # not C-contiguous
    with pytest.raises(ValueError):
        KMeans(2).fit(x)
    with pytest.raises(ValueError):
        KMeans(2).fit(np.zeros(6))


def test_kmeans_rejects_bad_parameters_and_non_finite_input():
    with pytest.raises(ValueError):
        KMeans(2, init="nonsense")
    with pytest.raises(ValueError):
        KMeans(0).fit(blobs())
    with pytest.raises(ValueError):
        KMeans(20).fit(blobs())
    bad = blobs()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        KMeans(2).fit(bad)


def test_unfitted_access_raises_runtime_error():
    km = KMeans(2)
    with pytest.raises(RuntimeError):
        km.predict(blobs())
    with pytest.raises(RuntimeError):
        _ = km.labels_


def test_pca_round_trip_and_explained_variance():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 3))
    pca = PCA(3)
    z = pca.fit_transform(x)
    # Outputs are lists; inputs must be arrays, so convert explicitly.
    back = pca.inverse_transform(np.array(z))
    assert np.allclose(np.array(back), x, atol=1e-9)
    ratio = pca.explained_variance_ratio_
    assert abs(sum(ratio) - 1.0) < 1e-9
    assert ratio == sorted(ratio, reverse=True)


def test_pca_float32_agrees_with_float64():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(30, 3))
    a = PCA(2)
    a.fit(x)
    b = PCA(2)
    b.fit(x.astype(np.float32))
    for ra, rb in zip(a.components_, b.components_):
        for va, vb in zip(ra, rb):
            assert abs(va - vb) < 1e-4


def test_pca_errors():
    with pytest.raises(ValueError):
        PCA(0).fit(np.zeros((5, 3)))
    with pytest.raises(ValueError):
        PCA(4).fit(np.zeros((5, 3)))
    with pytest.raises(RuntimeError):
        PCA(1).transform(np.zeros((2, 3)))


def test_fit_returns_self_so_calls_chain():
    x = blobs()
    labels = KMeans(2, seed=1).fit(x).predict(x)
    assert canonical(labels) == [0, 0, 0, 1, 1, 1]
    z = PCA(2).fit(x).transform(x)
    assert len(z) == 6
