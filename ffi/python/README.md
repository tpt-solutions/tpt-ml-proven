# tpt-ml-proven (Python)

Python bindings for KMeans and PCA, built with PyO3 and maturin as `abi3` wheels
(Python 3.9 and later).

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
