"""NumPy at the boundary, n_features_in_, fit_predict on the clusterers, and array-like inputs."""

import numpy as np
import pytest

from tpt_ml_proven import DBSCAN, HDBSCAN, PCA, KMeans, Ridge, StandardScaler, DecisionTreeRegressor

X = np.arange(40.0).reshape(10, 4)
Y = X[:, 0] * 2.0


def test_predictions_and_transforms_are_numpy_arrays():
    assert isinstance(Ridge().fit(X, Y).predict(X), np.ndarray)
    assert isinstance(PCA(2).fit(X).transform(X), np.ndarray)
    assert isinstance(KMeans(2).fit(X).predict(X), np.ndarray)
    assert isinstance(StandardScaler().fit_transform(X), np.ndarray)


def test_list_inputs_are_accepted():
    as_list = X.tolist()
    np.testing.assert_allclose(Ridge().fit(as_list, Y.tolist()).predict(as_list), Ridge().fit(X, Y).predict(X))


def test_pandas_frames_and_series_are_accepted():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(X, columns=list("abcd"))
    target = pd.Series(Y)
    np.testing.assert_allclose(Ridge().fit(frame, target).predict(frame), Ridge().fit(X, Y).predict(X))


def test_n_features_in_is_absent_before_fit_and_set_after():
    model = DecisionTreeRegressor(3)
    assert not hasattr(model, "n_features_in_")
    model.fit(X, Y.tolist())
    assert model.n_features_in_ == 4


def test_fit_predict_on_the_clusterers_returns_their_labels():
    dbscan = DBSCAN(1.5, 2)
    assert list(dbscan.fit_predict(X)) == list(DBSCAN(1.5, 2).fit(X).labels_)
    hdb = HDBSCAN(3)
    assert list(hdb.fit_predict(X)) == list(HDBSCAN(3).fit(X).labels_)


def test_results_match_the_list_form_element_for_element():
    model = Ridge().fit(X, Y)
    assert [float(v) for v in model.predict(X)] == pytest.approx(list(model.predict(X)))
