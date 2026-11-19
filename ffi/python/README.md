# tpt-ml-proven (Python)

Python bindings for KMeans, PCA and the Phase 8 estimators, built with PyO3 and maturin as
`abi3` wheels (Python 3.9 and later).

```sh
pip install maturin
maturin develop --release        # from ffi/python, installs into the active environment
pytest tests
```

```python
import numpy as np
from tpt_ml_proven import KMeans, PCA

x = np.random.default_rng(0).normal(size=(500, 4))   # float32 or float64, C-contiguous
labels = KMeans(8, seed=0).fit(x).predict(x)
z = PCA(2).fit(x).transform(x)
```

- Inputs are read in place when the dtype matches, with no copy. Use
  `np.ascontiguousarray` for other layouts.
- `fit` releases the GIL. Do not modify the array while it runs.
- Results are returned as Python lists.
- Errors raise `ValueError` for bad parameters or shapes, and `RuntimeError` before `fit`.

### Phase 8 estimators

`DBSCAN`, `HDBSCAN`, `TruncatedSVD`, `NMF`, `Ridge`, `Lasso`, `ElasticNet`,
`LogisticRegression` (binary or multinomial, `C` as in scikit-learn) and `StumpForest`.
These copy the input into `float64` before fitting, so `float32` and `float64` inputs give
the same result. Targets are passed as a sequence of numbers. Labels come back as numbers.

```python
from tpt_ml_proven import DBSCAN, LogisticRegression

labels = DBSCAN(eps=0.5, min_samples=5).fit(x).labels_
proba = LogisticRegression(C=1.0).fit(x, y).predict_proba(x)
```


Not yet exposed in Python: the scalers, encoders and imputers, the tree models
(`DecisionTreeRegressor`, `GradientBoostingRegressor`), and model files (save and load).
The Rust crates have all of these; the model-file format is in
[docs/model-format.md](../../docs/model-format.md).
