"""NumPy at the boundary of the estimators.

The compiled estimators read 2-D float arrays through NumPy's array interface, and return
plain lists. This module wraps their methods so that:

* inputs that are not NumPy arrays (a pandas or polars frame, a list of rows) are converted to
  contiguous float64 arrays first, and targets to lists of floats;
* array-valued results (predictions, transforms, probabilities) come back as NumPy arrays.

NumPy is optional. Without it, the methods behave as the bare extension does: lists in and out.
"""

import functools

try:  # NumPy is optional at run time.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only without NumPy installed
    _np = None

# Methods whose result is a 1-D or 2-D array.
_ARRAY_RESULTS = ("predict", "transform", "fit_transform", "predict_proba", "fit_predict", "inverse_transform")
# Methods that take a feature matrix as their first argument.
_MATRIX_INPUTS = ("fit", "predict", "transform", "fit_transform", "predict_proba", "fit_predict", "inverse_transform", "score")
# Methods that take a target as their second argument.
_TARGET_INPUTS = ("fit", "score")


def _as_matrix(x):
    """A float64 C-contiguous array view of `x`, unless it already is one."""
    if _np is None or hasattr(x, "__array_interface__"):
        return x
    return _np.ascontiguousarray(x, dtype=_np.float64)


def _as_targets(y):
    """A list of floats for a target sequence, or `y` unchanged when NumPy is absent."""
    if y is None or _np is None:
        return y
    return _np.asarray(y, dtype=_np.float64).ravel().tolist()


def _as_result(value):
    """NumPy array for a result, or the value unchanged when NumPy is absent."""
    if _np is None:
        return value
    return _np.asarray(value)


def _wrap(name, method):
    takes_matrix = name in _MATRIX_INPUTS
    takes_target = name in _TARGET_INPUTS
    returns_array = name in _ARRAY_RESULTS

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        args = list(args)
        if takes_matrix and args:
            args[0] = _as_matrix(args[0])
        if takes_target and len(args) > 1:
            args[1] = _as_targets(args[1])
        if "y" in kwargs and takes_target:
            kwargs["y"] = _as_targets(kwargs["y"])
        result = method(self, *args, **kwargs)
        return _as_result(result) if returns_array else result

    return wrapper


def install(classes):
    """Wraps the array-facing methods of each estimator class, in place."""
    for cls in classes:
        for name in set(_MATRIX_INPUTS) | set(_ARRAY_RESULTS) | set(_TARGET_INPUTS):
            method = cls.__dict__.get(name)
            if method is not None:
                setattr(cls, name, _wrap(name, method))
