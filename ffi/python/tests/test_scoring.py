"""score and fit_predict on the Python estimators, checked against the reference formulas."""

import numpy as np
import pytest

from tpt_ml_proven import KMeans, ElasticNet, Lasso, LogisticRegression, Ridge, StumpForest


def data():
    x = np.array([[float(i % 6), float((i * 5) % 7)] for i in range(40)], dtype=np.float64)
    y = 1.5 * x[:, 0] - 0.5 * x[:, 1] + 0.25
    labels = (x[:, 0] > 2.5).astype(np.float64)
    return x, y, labels


def r2(truth, pred):
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return 1.0 - ((truth - pred) ** 2).sum() / ((truth - truth.mean()) ** 2).sum()


@pytest.mark.parametrize("make", [lambda: Ridge(1e-6), lambda: Lasso(1e-6), lambda: ElasticNet(1e-6, 0.5)])
def test_regressor_score_is_r_squared_of_its_predictions(make):
    x, y, _ = data()
    model = make().fit(x, y.tolist())
    assert model.score(x, y.tolist()) == pytest.approx(r2(y, model.predict(x)), abs=1e-12)
    assert model.score(x, y.tolist()) > 0.999


def test_stump_forest_score_is_accuracy():
    x, _, labels = data()
    model = StumpForest(15, seed=1).fit(x, labels.tolist())
    pred = np.asarray(model.predict(x))
    assert model.score(x, labels.tolist()) == pytest.approx(float(np.mean(pred == labels)))


def test_logistic_score_is_accuracy():
    x, _, labels = data()
    model = LogisticRegression(10.0).fit(x, labels.tolist())
    pred = np.asarray(model.predict(x))
    assert model.score(x, labels.tolist()) == pytest.approx(float(np.mean(pred == labels)))


def test_kmeans_fit_predict_equals_fit_then_predict():
    x, _, _ = data()
    a = KMeans(3, seed=5).fit_predict(x)
    b = KMeans(3, seed=5).fit(x).predict(x)
    assert list(a) == list(b)


def test_score_rejects_mismatched_targets():
    x, y, _ = data()
    model = Ridge(1.0).fit(x, y.tolist())
    with pytest.raises(ValueError):
        model.score(x, y[:-1].tolist())
