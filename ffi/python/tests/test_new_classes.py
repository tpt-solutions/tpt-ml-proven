"""The scalers, encoders, imputer, trees and forests in Python: against scikit-learn where the
algorithm is exact, and by properties where it is randomised."""

import pickle

import numpy as np
import pytest

from tpt_ml_proven import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    GradientBoostingRegressor,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RandomForestClassifier,
    RandomForestRegressor,
    RobustScaler,
    SimpleImputer,
    StandardScaler,
)

RNG = np.random.default_rng(7)
X = RNG.normal(size=(60, 3))
Y = 2.0 * X[:, 0] - X[:, 1] + 0.1 * RNG.normal(size=60)
LABELS = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float64)
CATS = np.array([[float(i % 3), float((i * 2) % 4)] for i in range(24)])


# ---------------------------------------------------------------- against scikit-learn

def test_standard_scaler_matches_scikit_learn():
    sk = pytest.importorskip("sklearn.preprocessing")
    np.testing.assert_allclose(
        StandardScaler().fit(X).transform(X),
        sk.StandardScaler().fit(X).transform(X),
        rtol=1e-12, atol=1e-12,
    )


def test_minmax_and_robust_scalers_match_scikit_learn():
    sk = pytest.importorskip("sklearn.preprocessing")
    np.testing.assert_allclose(MinMaxScaler().fit(X).transform(X), sk.MinMaxScaler().fit(X).transform(X), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(RobustScaler().fit(X).transform(X), sk.RobustScaler().fit(X).transform(X), rtol=1e-10, atol=1e-10)


def test_one_hot_encoder_matches_scikit_learn():
    sk = pytest.importorskip("sklearn.preprocessing")
    ours = np.asarray(OneHotEncoder().fit(CATS).transform(CATS))
    theirs = sk.OneHotEncoder(sparse_output=False).fit(CATS).transform(CATS)
    np.testing.assert_array_equal(ours, theirs)


def test_ordinal_encoder_matches_scikit_learn():
    sk = pytest.importorskip("sklearn.preprocessing")
    ours = np.asarray(OrdinalEncoder().fit(CATS).transform(CATS))
    theirs = sk.OrdinalEncoder().fit(CATS).transform(CATS)
    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent", "constant"])
def test_simple_imputer_matches_scikit_learn(strategy):
    sk = pytest.importorskip("sklearn.impute")
    holey = X.copy()
    holey[::5, 0] = np.nan
    holey[::7, 2] = np.nan
    ours = np.asarray(SimpleImputer(strategy=strategy, fill_value=-9.0).fit(holey).transform(holey))
    theirs = sk.SimpleImputer(strategy=strategy, fill_value=-9.0).fit(holey).transform(holey)
    np.testing.assert_allclose(ours, theirs, rtol=1e-12)


def test_decision_tree_regressor_matches_scikit_learn():
    sk = pytest.importorskip("sklearn.tree")
    ours = DecisionTreeRegressor(4, min_samples_leaf=2).fit(X, Y.tolist())
    theirs = sk.DecisionTreeRegressor(max_depth=4, min_samples_leaf=2, random_state=0).fit(X, Y)
    np.testing.assert_allclose(ours.predict(X), theirs.predict(X), rtol=1e-9, atol=1e-12)


def test_decision_tree_classifier_matches_scikit_learn():
    sk = pytest.importorskip("sklearn.tree")
    for criterion in ["gini", "entropy"]:
        ours = DecisionTreeClassifier(3, criterion=criterion).fit(X, LABELS.tolist())
        theirs = sk.DecisionTreeClassifier(max_depth=3, criterion=criterion, random_state=0).fit(X, LABELS)
        assert list(ours.predict(X)) == list(theirs.predict(X)), criterion


# ---------------------------------------------------------------- by properties

def test_gradient_boosting_fits_the_signal():
    model = GradientBoostingRegressor(60, learning_rate=0.2, max_depth=3, seed=1).fit(X, Y.tolist())
    assert model.score(X, Y.tolist()) > 0.9


def test_random_forests_learn_and_repeat_with_a_seed():
    reg = RandomForestRegressor(30, max_depth=6, seed=2).fit(X, Y.tolist())
    assert reg.score(X, Y.tolist()) > 0.8
    clf = RandomForestClassifier(30, max_depth=5, seed=2).fit(X, LABELS.tolist())
    assert clf.score(X, LABELS.tolist()) > 0.9
    again = RandomForestClassifier(30, max_depth=5, seed=2).fit(X, LABELS.tolist())
    assert list(clf.predict(X)) == list(again.predict(X))


def test_classifier_classes_are_exposed():
    clf = DecisionTreeClassifier(2).fit(X, LABELS.tolist())
    assert list(clf.classes_) == [0.0, 1.0]


# ---------------------------------------------------------------- the shared protocol

NEW = [
    lambda: StandardScaler(),
    lambda: MinMaxScaler(),
    lambda: RobustScaler(),
    lambda: OneHotEncoder(),
    lambda: OrdinalEncoder(),
    lambda: SimpleImputer("median", fill_value=1.5),
    lambda: DecisionTreeRegressor(4, min_samples_leaf=2),
    lambda: DecisionTreeClassifier(3, criterion="entropy"),
    lambda: GradientBoostingRegressor(10, learning_rate=0.2, subsample=0.8, seed=3),
    lambda: RandomForestRegressor(5, max_depth=3, seed=4),
    lambda: RandomForestClassifier(5, max_depth=3, criterion="entropy", seed=5),
]


def _fit(model):
    if isinstance(model, (DecisionTreeRegressor, GradientBoostingRegressor, RandomForestRegressor)):
        return model.fit(X, Y.tolist())
    if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier)):
        return model.fit(X, LABELS.tolist())
    return model.fit(X)


def _output(model):
    if hasattr(model, "transform"):
        return np.asarray(model.transform(X))
    return np.asarray(model.predict(X))


@pytest.mark.parametrize("make", NEW)
def test_parameters_clone_and_pickle_round_trip(make, tmp_path):
    model = make()
    params = model.get_params()
    clone = model.__sklearn_clone__()
    assert clone.get_params() == params
    assert not clone.__sklearn_is_fitted__()
    fitted = _fit(model)
    assert fitted.__sklearn_is_fitted__()
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(_output(restored), _output(fitted))
    path = tmp_path / "m.tpml"
    fitted.save(str(path))
    loaded = type(fitted).load(str(path))
    np.testing.assert_array_equal(_output(loaded), _output(fitted))


@pytest.mark.parametrize("make", NEW)
def test_repr_names_the_class(make):
    model = make()
    assert repr(model).startswith(type(model).__name__ + "(")


def test_set_params_on_a_tree_takes_effect_on_refit():
    model = DecisionTreeRegressor(1).fit(X, Y.tolist())
    shallow = model.predict(X)
    model.set_params(max_depth=6).fit(X, Y.tolist())
    assert not np.allclose(model.predict(X), shallow)


def test_an_imputer_rejects_an_unknown_strategy():
    with pytest.raises(ValueError):
        SimpleImputer("nearest")


def test_a_tree_rejects_an_unknown_criterion():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(3, criterion="chaos")
