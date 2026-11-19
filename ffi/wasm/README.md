# tpt-ml-proven (WebAssembly)

WebAssembly bindings for the library's estimators, built with wasm-bindgen. They cover
clustering, decomposition, linear models, the stump forest, trees and forests, and the
preprocessing transformers.

## Build

```sh
node scripts/build.mjs      # needs wasm-pack; writes pkg/ (nodejs), pkg-web/ and pkg-bundler/
node --test tests/bindings.test.mjs tests/web-target.test.mjs   # the nodejs build and the web build
```

Each output directory has a `package.json` written from `package.template.json`, with the
version taken from `Cargo.toml`.

| Directory | Target | Use |
| --- | --- | --- |
| `pkg/` | `nodejs` | `require()` from Node |
| `pkg-web/` | `web` | an ES module in a browser; call the default export `init()` first |
| `pkg-bundler/` | `bundler` | an ES module for webpack, Vite and similar |

`demo/index.html` loads the `web` build and clusters random points. Serve the folder with any
static server (browsers do not load wasm from `file://`). The demo has not been run in a browser
from this repository; the `web` build is tested under Node.

## Conventions

- **Options object.** Every constructor takes one optional object with camelCase keys:
  `new KMeansModel({ nClusters: 3, seed: 7 })`. Missing keys take their defaults. A wrong type or
  a missing required key is an error that names the key.
- **Seeds and counts are numbers**, not BigInts, and must be non-negative integers no larger than
  `Number.MAX_SAFE_INTEGER`.
- **Matrices are flat, row-major, with `cols` features per row.** Pass the data and `cols`:
  `model.fit(data, cols)`. `Float64Array` is the default; `fitF32` takes a `Float32Array` on the
  KMeans and PCA models.
- **Targets** are `Float64Array`s with one value per row. Class labels come back as
  `Float64Array`s, KMeans labels as `Uint32Array`, and DBSCAN and HDBSCAN labels as `Int32Array`
  (`-1` is noise).
- **Errors** are thrown as JavaScript `Error` objects. Calling a method before `fit` throws
  "not fitted".

```js
const { KMeansModel, RidgeModel, DecisionTreeClassifierModel } = require("./pkg/tpt_ml_proven_wasm.js");

const data = new Float64Array([0, 0, 0.1, 0, 10, 10, 10.1, 10]); // 4 rows x 2 features
const km = new KMeansModel({ nClusters: 2, seed: 1 });
km.fit(data, 2);
km.labels;                                  // Uint32Array, one label per row
km.centers;                                 // Float64Array, n_clusters x n_features

const x = new Float64Array([0, 1, 2, 3, 4, 5]);
const y = new Float64Array([1, 3, 5]);
const ridge = new RidgeModel({ alpha: 0.1 });
ridge.fit(x, 1, y);
ridge.predict(x, 1);                        // Float64Array
ridge.score(x, 1, y);                       // R²
```

## Estimators

| Family | Classes and their options (defaults in brackets) |
| --- | --- |
| Clustering | `KMeansModel` (`nClusters`, `maxIter` 300, `tol` 1e-4, `init` `"k-means++"`, `seed` 0); `DbscanModel` (`eps`, `minSamples`); `HdbscanModel` (`minClusterSize` 5, `minSamples`) |
| Decomposition | `PcaModel` (`nComponents`); `TruncatedSvdModel` (`nComponents`); `NmfModel` (`nComponents`, `maxIter` 200, `tol` 1e-4, `seed` 0) |
| Linear | `RidgeModel` (`alpha` 1); `LassoModel` (`alpha` 1, `maxIter` 1000, `tol` 1e-4); `ElasticNetModel` (`alpha` 1, `l1Ratio` 0.5, `maxIter`, `tol`); `LogisticRegressionModel` (`C` 1, `maxIter` 100, `tol`) |
| Ensembles | `StumpForestModel` (`nEstimators` 100, `seed` 0) |
| Preprocessing | `StandardScalerModel`, `MinMaxScalerModel`, `RobustScalerModel`, `OneHotEncoderModel`, `OrdinalEncoderModel`; `SimpleImputerModel` (`strategy` `"mean"`, `"median"`, `"most_frequent"` or `"constant"`; `fillValue` 0) |
| Trees and forests | `DecisionTreeRegressorModel` (`maxDepth` 3, `minSamplesLeaf` 1); `DecisionTreeClassifierModel` (plus `criterion` `"gini"` or `"entropy"`); `GradientBoostingRegressorModel` (`nEstimators` 100, `learningRate` 0.1, `maxDepth` 3, `minSamplesLeaf` 1, `subsample` 1, `maxFeatures`, `seed` 0); `RandomForestRegressorModel` (`nEstimators` 100, `maxDepth` 10, `minSamplesLeaf` 1, `maxFeatures`, `seed` 0); `RandomForestClassifierModel` (as the regressor, plus `criterion`) |

Transformers have `fit`, `transform` and `fitTransform`, and an `outputCols` getter once fitted.
Supervised models have `fit`, `predict` and `score` (R² for regressors, accuracy for
classifiers). The classifiers and the logistic model expose `classes`.

## Not yet in WebAssembly

Model files (`toBytes`/`fromBytes`) are not exposed here. The Python package and the Rust crates
have them. The `f32` path covers KMeans and PCA only.
