# tpt-ml-proven (WebAssembly)

WebAssembly bindings for KMeans and PCA, built with wasm-bindgen.

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
