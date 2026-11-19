"""Interoperability with scikit-learn: pipelines, cloning, cross-validation and grid search."""

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import GridSearchCV, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from tpt_ml_proven import KMeans, Lasso, Ridge  # noqa: E402

RNG = np.random.default_rng(3)
X = RNG.normal(size=(60, 4))
Y = 2.0 * X[:, 0] - X[:, 2] + 0.1 * RNG.normal(size=60)


def test_a_pipeline_with_a_scikit_learn_scaler_matches_the_manual_steps():
    pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(0.5))]).fit(X, Y)
    manual = Ridge(0.5).fit(StandardScaler().fit_transform(X), Y)
    np.testing.assert_allclose(pipe.predict(X), manual.predict(StandardScaler().fit(X).transform(X)), rtol=1e-12)


def test_clone_gives_an_unfitted_estimator_with_the_same_parameters():
    original = Lasso(0.2, max_iter=77).fit(X, Y)
    copy = clone(original)
    assert copy.get_params() == original.get_params()
    with pytest.raises(RuntimeError):
        copy.to_bytes()


def test_cross_validation_scores_a_regressor_through_its_score_method():
    scores = cross_val_score(Ridge(0.1), X, Y, cv=4)
    assert len(scores) == 4
    assert all(s > 0.9 for s in scores)


def test_grid_search_selects_a_penalty():
    search = GridSearchCV(Ridge(), {"alpha": [1e-6, 1.0, 1e3]}, cv=3).fit(X, Y)
    assert search.best_params_["alpha"] == 1e-6
    assert search.best_estimator_.coef_ is not None


def test_kmeans_works_in_a_pipeline_after_scaling():
    pipe = Pipeline([("scale", StandardScaler()), ("km", KMeans(3, seed=1))]).fit(X)
    labels = pipe.predict(X)
    assert len(set(labels)) == 3
