"""The single-tree settings: the split size, the leaf limit and feature importances.

On continuous data there are no ties, so the trees should agree with scikit-learn's exactly.
"""

import numpy as np
import pytest

from tpt_ml_proven import DecisionTreeClassifier, DecisionTreeRegressor

skb = pytest.importorskip("sklearn.base")
sk = pytest.importorskip("sklearn.tree")


def regression_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 3))
    y = 2.0 * x[:, 0] - x[:, 1] ** 2 + 0.1 * rng.normal(size=120)
    return x, y


def classification_data():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(120, 3))
    y = (x[:, 0] > 0).astype(float) + (x[:, 2] > 0.5).astype(float)
    return x, y


def test_regressor_matches_scikit_learn_with_the_new_limits():
    x, y = regression_data()
    ours = DecisionTreeRegressor(
        6, min_samples_leaf=3, min_samples_split=8, max_leaf_nodes=10
    ).fit(x, y.tolist())
    theirs = sk.DecisionTreeRegressor(
        max_depth=6, min_samples_leaf=3, min_samples_split=8, max_leaf_nodes=10, random_state=0
    ).fit(x, y)
    assert np.allclose(ours.predict(x), theirs.predict(x), atol=1e-9)
    assert np.allclose(ours.feature_importances_, theirs.feature_importances_, atol=1e-9)


def test_classifier_matches_scikit_learn_with_the_new_limits():
    x, y = classification_data()
    ours = DecisionTreeClassifier(
        5, min_samples_split=6, max_leaf_nodes=8, criterion="entropy"
    ).fit(x, y.tolist())
    theirs = sk.DecisionTreeClassifier(
        max_depth=5, min_samples_split=6, max_leaf_nodes=8, criterion="entropy", random_state=0
    ).fit(x, y)
    assert np.array_equal(np.asarray(ours.predict(x)), theirs.predict(x))
    assert np.allclose(ours.feature_importances_, theirs.feature_importances_, atol=1e-9)


def test_importances_are_unavailable_before_fit():
    with pytest.raises(AttributeError):
        _ = DecisionTreeRegressor(3).feature_importances_
    with pytest.raises(AttributeError):
        _ = DecisionTreeClassifier(3).feature_importances_


def test_new_settings_are_parameters_that_clone_keeps():
    model = DecisionTreeRegressor(4, min_samples_split=5, max_leaf_nodes=6)
    params = model.get_params()
    assert params["min_samples_split"] == 5
    assert params["max_leaf_nodes"] == 6
    clone = skb.clone(model)
    assert clone.get_params() == params
    model.set_params(max_leaf_nodes=None)
    assert model.get_params()["max_leaf_nodes"] is None


def test_a_model_with_new_settings_round_trips_through_bytes():
    x, y = regression_data()
    model = DecisionTreeRegressor(5, min_samples_split=4, max_leaf_nodes=7).fit(x, y.tolist())
    loaded = DecisionTreeRegressor.from_bytes(model.to_bytes())
    assert loaded.get_params() == model.get_params()
    assert np.array_equal(loaded.predict(x), model.predict(x))
    assert np.array_equal(loaded.feature_importances_, model.feature_importances_)


def test_an_older_model_file_loads_with_the_defaults():
    # A model without the new settings, saved by the default constructor, loads with the defaults.
    x, y = regression_data()
    model = DecisionTreeRegressor(3).fit(x, y.tolist())
    loaded = DecisionTreeRegressor.from_bytes(model.to_bytes())
    assert loaded.get_params()["min_samples_split"] == 2
    assert loaded.get_params()["max_leaf_nodes"] is None
    assert np.array_equal(loaded.predict(x), model.predict(x))


def test_invalid_limits_are_rejected():
    x, y = regression_data()
    with pytest.raises(ValueError):
        DecisionTreeRegressor(3, min_samples_split=1).fit(x, y.tolist())
    with pytest.raises(ValueError):
        DecisionTreeClassifier(3, max_leaf_nodes=1).fit(x, classification_data()[1].tolist())
