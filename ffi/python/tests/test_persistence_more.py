"""Save, load and pickle for the clusterers and for logistic regression, including the label
values of a binary model, which the file must keep."""

import pickle

import numpy as np
import pytest

from tpt_ml_proven import DBSCAN, HDBSCAN, LogisticRegression

RNG = np.random.default_rng(11)
BLOBS = np.vstack([RNG.normal(0.0, 0.2, size=(20, 2)), RNG.normal(4.0, 0.2, size=(20, 2))])
SCORE = BLOBS[:, 0] - BLOBS[:, 1]


def check(model, data, predict, tmp_path):
    before = list(predict(model, data))
    assert list(predict(type(model).from_bytes(model.to_bytes()), data)) == before
    path = tmp_path / "m.tpml"
    model.save(str(path))
    assert list(predict(type(model).load(str(path)), data)) == before
    assert list(predict(pickle.loads(pickle.dumps(model)), data)) == before


def test_dbscan_round_trips_through_bytes_files_and_pickle(tmp_path):
    model = DBSCAN(0.6, 4).fit(BLOBS)
    check(model, BLOBS, lambda m, d: m.labels_, tmp_path)
    assert model.get_params() == {"eps": 0.6, "min_samples": 4}
    loaded = DBSCAN.from_bytes(model.to_bytes())
    assert loaded.get_params() == model.get_params()
    assert list(loaded.core_sample_indices_) == list(model.core_sample_indices_)


def test_hdbscan_round_trips_through_bytes_files_and_pickle(tmp_path):
    model = HDBSCAN(5).fit(BLOBS)
    check(model, BLOBS, lambda m, d: m.labels_, tmp_path)
    assert model.get_params() == {"min_cluster_size": 5, "min_samples": None}


def test_a_binary_logistic_model_keeps_its_two_labels(tmp_path):
    labels = np.where(SCORE > 0, 7.0, 3.0)
    model = LogisticRegression(2.0).fit(BLOBS, labels.tolist())
    loaded = LogisticRegression.from_bytes(model.to_bytes())
    assert list(loaded.classes_) == [3.0, 7.0]
    assert list(loaded.predict(BLOBS)) == list(model.predict(BLOBS))
    assert set(loaded.predict(BLOBS)) <= {3.0, 7.0}
    check(model, BLOBS, lambda m, d: m.predict(d), tmp_path)


def test_a_multiclass_logistic_model_round_trips(tmp_path):
    labels = (np.arange(len(BLOBS)) % 3).astype(float)  # three classes, whatever the geometry
    model = LogisticRegression(1.0).fit(BLOBS, labels.tolist())
    loaded = LogisticRegression.from_bytes(model.to_bytes())
    assert list(loaded.classes_) == list(model.classes_)
    check(model, BLOBS, lambda m, d: m.predict(d), tmp_path)


def test_a_loaded_logistic_model_keeps_its_settings():
    model = LogisticRegression(2.5, max_iter=40, tol=1e-3).fit(BLOBS, (SCORE > 0).astype(float).tolist())
    loaded = LogisticRegression.from_bytes(model.to_bytes())
    assert loaded.get_params() == {"C": 2.5, "max_iter": 40, "tol": 1e-3}


def test_unfitted_clusterers_refuse_to_save():
    with pytest.raises(RuntimeError):
        DBSCAN(0.5, 3).to_bytes()
    with pytest.raises(RuntimeError):
        HDBSCAN(4).to_bytes()
