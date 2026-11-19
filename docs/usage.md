# Usage guide

Examples for each language binding. The estimators follow the same pattern everywhere: build
the model with its parameters, fit it on the data, then predict, transform or read the fitted
attributes. Each algorithm's contract is in [docs/contracts/](contracts/README.md).

Which estimators each binding exposes:

| Estimator | Rust | Python | WASM | C ABI | Go |
| --- | --- | --- | --- | --- | --- |
| KMeans, PCA | yes | yes | yes | yes | yes |
| DBSCAN, HDBSCAN | yes | yes | yes | yes | no |
| TruncatedSVD, NMF | yes | yes | yes | yes | no |
| Ridge, Lasso, ElasticNet | yes | yes | yes | yes | no |
| LogisticRegression (binary and multinomial) | yes | yes | yes | yes | no |
| StumpForest | yes | yes | yes | yes | no |
| Scalers, encoders, imputers | yes | no | no | no | no |
| DecisionTreeRegressor, GradientBoostingRegressor | yes | no | no | no | no |
| Model files (save and load) | yes | no | no | no | no |

## Rust

The core traits are in `tpt-ml-proven-core`. Data is a row-major view, `DenseView`, over a
slice. Fitting is `fit`, and `predict` or `transform` use the fitted state.

```rust
use tpt_ml_proven_clustering::KMeans;
use tpt_ml_proven_core::{DenseView, Predictor, UnsupervisedEstimator};

let data = [0.0_f64, 0.0, 0.1, 0.0, 10.0, 10.0, 10.1, 10.0]; // 4 rows x 2 columns
let mut km = KMeans::<f64>::new(2).with_seed(1);
UnsupervisedEstimator::fit(&mut km, DenseView::new(&data, 4, 2)?)?;
let labels = km.labels().unwrap();
```

Supervised estimators take a target slice:

```rust
use tpt_ml_proven_core::{DenseView, Predictor, SupervisedEstimator};
use tpt_ml_proven_linear::Ridge;

let mut model = Ridge::<f64>::new(0.1);
SupervisedEstimator::fit(&mut model, DenseView::new(&x, rows, cols)?, &y[..])?;
let predictions = model.predict(DenseView::new(&x, rows, cols)?)?;
```

Model files store a fitted model in the version 1 format, and load it back:

```rust
use tpt_ml_proven_linear::Ridge;

let bytes = model.to_bytes()?;            // Vec<u8>, see docs/model-format.md
std::fs::write("ridge.tpml", &bytes)?;
let loaded = Ridge::<f64>::from_bytes(&std::fs::read("ridge.tpml")?)?;
```

Tree models come from the trees crate:

```rust
use tpt_ml_proven_trees::GradientBoostingRegressor;

let mut model = GradientBoostingRegressor::new(100, 3).with_learning_rate(0.1);
model.fit(DenseView::new(&x, rows, cols)?, &y[..])?;
let curve = model.train_loss().unwrap();  // never increases, see the contract
```

Errors are `tpt_ml_proven_core::error::Error`. Parameter, shape and finiteness problems are
reported before any work is done, and a failed fit leaves the estimator unfitted.

## Python

```python
import numpy as np
from tpt_ml_proven import KMeans, PCA, DBSCAN, Ridge, LogisticRegression

x = np.random.default_rng(0).normal(size=(500, 4))     # float32 or float64

labels = KMeans(8, seed=0).fit(x).predict(x)           # fit returns the estimator
z = PCA(2).fit(x).transform(x)

clusters = DBSCAN(eps=0.5, min_samples=5).fit(x).labels_   # -1 marks noise

y = (x[:, 0] > 0).astype(float).tolist()               # targets are plain sequences
proba = LogisticRegression(C=1.0).fit(x, y).predict_proba(x)
```

- `fit` releases the GIL. Do not modify the array while it runs.
- Results are returned as Python lists.
- Errors raise `ValueError` for bad parameters or shapes, and `RuntimeError` before `fit`.

## WebAssembly (Node)

Data is a flat, row-major `Float64Array`, with the column count passed alongside.

```js
const { KMeansModel, RidgeModel } = require("./pkg/tpt_ml_proven_wasm.js");

const data = new Float64Array([0, 0, 0.1, 0, 10, 10, 10.1, 10]);
const km = new KMeansModel(2, undefined, undefined, undefined, 1n); // seed is a BigInt
km.fit(data, 2);
const labels = km.labels;                                        // Uint32Array

const ridge = new RidgeModel(0.1);
ridge.fit(data, 2, Float64Array.from([0, 1, 10, 10]));
const predictions = ridge.predict(data, 2);
```

Build with `wasm-pack build --target nodejs --no-opt` from `ffi/wasm`.

## Go

```go
import mlproven "github.com/tpt-solutions/tpt-ml-proven/ffi/go"

km, err := mlproven.NewKMeans(2, mlproven.WithSeed(1))
if err != nil { /* handle */ }
defer km.Close()
if err := km.Fit(x); err != nil { /* handle */ }   // x is [][]float64, one slice per row
labels, _ := km.Labels()
```

Build the native library first, as described in [ffi/go/README.md](../ffi/go/README.md).

## C

The C ABI uses opaque handles, caller-allocated output buffers, and a status code from every
call. Each `*_new` has a matching `*_free`. The header is `ffi/c/include/tpt_ml_proven.h`.

```c
struct tpml_TpmlDbscan *db = tpml_dbscan_new(0.5, 5);
enum tpml_TpmlStatus st = tpml_dbscan_fit(db, data, rows, cols);
if (st != tpml_TpmlStatus_Ok) {
    fprintf(stderr, "%s\n", tpml_last_error_message());
}
int64_t labels[rows];
tpml_dbscan_labels(db, labels, rows);
tpml_dbscan_free(db);
```

The error message is valid until the next call on the same thread.
