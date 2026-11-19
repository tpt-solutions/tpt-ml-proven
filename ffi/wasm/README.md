# tpt-ml-proven (WebAssembly)

WebAssembly bindings for KMeans, PCA and the Phase 8 estimators, built with wasm-bindgen.

```sh
wasm-pack build --target nodejs --release --no-opt
node --test tests/bindings.test.mjs
```

```js
const { KMeansModel, PcaModel } = require("./pkg/tpt_ml_proven_wasm.js");

const data = new Float64Array([0, 0, 0.1, 0, 10, 10, 10.1, 10]); // rows x cols, row-major
const km = new KMeansModel(2, undefined, undefined, undefined, 1n); // seed is a BigInt
km.fit(data, 2);
const labels = km.labels;              // Uint32Array
const p = new PcaModel(1);
p.fit(data, 2);
const z = p.transform(data, 2);        // flat rows x n_components
```

- Data is flat and row-major. The column count is passed alongside.
- `fitF32` / `Float32Array` keeps the model in single precision.
- Errors are thrown as JavaScript `Error` objects.
- Seeds are `u64`, so they are passed as BigInts.

## Phase 8 estimators

`DbscanModel`, `HdbscanModel`, `TruncatedSvdModel`, `NmfModel`, `RidgeModel`, `LassoModel`,
`ElasticNetModel`, `LogisticRegressionModel` (binary or multinomial) and `StumpForestModel`.
They run in `f64`. Targets are `Float64Array`s, labels come back as `Int32Array`
(`-1` for noise), and matrices are flat and row-major.

```js
const { DbscanModel, LogisticRegressionModel } = require("./pkg/tpt_ml_proven_wasm.js");

const db = new DbscanModel(0.5, 5);
db.fit(data, 2);
const labels = db.labels;                       // Int32Array, -1 for noise

const lr = new LogisticRegressionModel(1.0);
lr.fit(data, 2, Float64Array.from([0, 0, 0, 1, 1, 1]));
const proba = lr.predictProba(data, 2);         // flat, one row per sample
```


Not yet exposed in WebAssembly: the scalers, encoders and imputers, the tree models, and
model files (save and load). See [docs/model-format.md](../../docs/model-format.md) for the
format.
