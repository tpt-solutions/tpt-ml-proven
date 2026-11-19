"""save, load, to_bytes, from_bytes and pickle for the estimators that have a model file format."""

import pickle

import numpy as np
import pytest

from tpt_ml_proven import NMF, PCA, KMeans, ElasticNet, Lasso, Ridge, StumpForest, TruncatedSVD

# Two blobs in four dimensions, with a deterministic wobble so every fit is well posed.
X = [[float(i % 7) + 0.1 * j, float((i * 3) % 5), float(i % 4), 0.5 * j] for i in range(30) for j in range(1)]
Y = [sum(row) * 0.5 - row[0] for row in X]
LABELS = [0.0 if row[0] < 3.5 else 1.0 for row in X]
NON_NEGATIVE = [[abs(v) + 0.1 for v in row] for row in X]

# The bindings take 2-D C-contiguous NumPy arrays.
X = np.ascontiguousarray(X, dtype=np.float64)
NON_NEGATIVE = np.ascontiguousarray(NON_NEGATIVE, dtype=np.float64)


def roundtrip_via_file(model, tmp_path):
    path = tmp_path / "model.tpml"
    model.save(str(path))
    return type(model).load(str(path))


def check_round_trips(model, data, tmp_path, predict):
    before = predict(model, data)
    via_bytes = type(model).from_bytes(model.to_bytes())
    assert predict(via_bytes, data) == before
    assert predict(roundtrip_via_file(model, tmp_path), data) == before
    assert predict(pickle.loads(pickle.dumps(model)), data) == before


def transform(model, data):
    return np.asarray(model.transform(data)).tolist()


def labels_of(model, data):
    return list(model.predict(data))


def values_of(model, data):
    return list(model.predict(data))


@pytest.mark.parametrize(
    "make, fit_to, predict, data",
    [
        (lambda: KMeans(3, seed=2), None, labels_of, X),
        (lambda: PCA(2), None, transform, X),
        (lambda: TruncatedSVD(2), None, transform, NON_NEGATIVE),
        (lambda: NMF(2, seed=4), None, transform, NON_NEGATIVE),
        (lambda: Ridge(0.5), Y, values_of, X),
        (lambda: Lasso(0.05), Y, values_of, X),
        (lambda: ElasticNet(0.05, 0.4), Y, values_of, X),
        (lambda: StumpForest(7, seed=3), LABELS, values_of, X),
    ],
)
def test_fitted_models_round_trip_through_bytes_files_and_pickle(make, fit_to, predict, data, tmp_path):
    model = make()
    if fit_to is None:
        model.fit(data)
    else:
        model.fit(data, fit_to)
    check_round_trips(model, data, tmp_path, predict)


def test_a_loaded_kmeans_refits_with_its_saved_settings():
    original = KMeans(3, max_iter=50, tol=1e-3, init="random", seed=9, threads=2)
    original.fit(X)
    loaded = KMeans.from_bytes(original.to_bytes())
    assert loaded.fit(X).labels_ == original.labels_
    fresh = KMeans(3, max_iter=50, tol=1e-3, init="random", seed=9, threads=2).fit(X)
    assert loaded.labels_ == fresh.labels_


def test_a_loaded_ridge_refits_with_its_saved_penalty():
    original = Ridge(0.7).fit(X, Y)
    loaded = Ridge.from_bytes(original.to_bytes())
    assert loaded.fit(X, Y).coef_ == Ridge(0.7).fit(X, Y).coef_


def test_unfitted_models_refuse_to_save():
    with pytest.raises(RuntimeError):
        KMeans(2).to_bytes()
    with pytest.raises(RuntimeError):
        Ridge().to_bytes()


def test_a_file_of_another_kind_is_refused(tmp_path):
    ridge_bytes = Ridge(1.0).fit(X, Y).to_bytes()
    with pytest.raises(ValueError):
        KMeans.from_bytes(ridge_bytes)


def test_garbage_bytes_are_refused():
    with pytest.raises(ValueError):
        Ridge.from_bytes(b"not a model file, just text in bytes")
