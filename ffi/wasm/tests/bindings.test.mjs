// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions
// Behaviour of the WebAssembly bindings. Run with `node --test tests/` after
// `wasm-pack build --target nodejs --no-opt`.

import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { KMeansModel, PcaModel } = require("../pkg/tpt_ml_proven_wasm.js");

// Two blobs of three points each, row-major, 2 features.
const BLOBS = [0, 0, 0.1, 0, 0, 0.1, 10, 10, 10.1, 10, 10, 10.1];

function canonical(labels) {
  const seen = [];
  return labels.map((l) => {
    if (!seen.includes(l)) seen.push(l);
    return seen.indexOf(l);
  });
}

test("KMeans recovers blobs with f64 input", () => {
  const km = new KMeansModel(2, undefined, undefined, undefined, 1n);
  km.fit(new Float64Array(BLOBS), 2);
  assert.deepEqual(canonical(Array.from(km.labels)), [0, 0, 0, 1, 1, 1]);
  assert.ok(km.inertia < 0.1);
  assert.equal(km.centers.length, 4);
});

test("KMeans f32 path agrees with f64 on easy data", () => {
  const a = new KMeansModel(2, undefined, undefined, undefined, 1n);
  a.fit(new Float64Array(BLOBS), 2);
  const b = new KMeansModel(2, undefined, undefined, undefined, 1n);
  b.fitF32(new Float32Array(BLOBS), 2);
  assert.deepEqual(canonical(Array.from(a.labels)), canonical(Array.from(b.labels)));
});

test("KMeans predict and transform shapes", () => {
  const km = new KMeansModel(2, undefined, undefined, undefined, 1n);
  km.fit(new Float64Array(BLOBS), 2);
  assert.deepEqual(Array.from(km.predict(new Float64Array(BLOBS), 2)), Array.from(km.labels));
  const d = km.transform(new Float64Array(BLOBS), 2);
  assert.equal(d.length, 6 * 2);
  assert.ok(d.every((v) => v >= 0));
});

test("KMeans errors are thrown as JavaScript errors", () => {
  assert.throws(() => new KMeansModel(2, undefined, undefined, "bogus"), /init must be/);
  const km = new KMeansModel(2);
  assert.throws(() => km.predict(new Float64Array(BLOBS), 2), /not fitted/);
  assert.throws(() => km.fit(new Float64Array(BLOBS), 5), /multiple of cols/);
  assert.throws(() => new KMeansModel(20).fit(new Float64Array(BLOBS), 2), /exceeds/);
});

test("PCA round trip reconstructs a full-rank projection", () => {
  const data = new Float64Array([1, 2, 3, 4, 2, 1, 0, 5, 3, 3, 1, 2]);
  const p = new PcaModel(2);
  p.fit(data, 2);
  const z = p.transform(data, 2);
  const back = p.inverseTransform(z, 2);
  for (let i = 0; i < data.length; i++) {
    assert.ok(Math.abs(back[i] - data[i]) < 1e-9, `${back[i]} vs ${data[i]}`);
  }
  const ratio = Array.from(p.explainedVarianceRatio);
  assert.ok(Math.abs(ratio.reduce((a, b) => a + b, 0) - 1) < 1e-9);
});

test("PCA f32 agrees with f64 to single precision", () => {
  const data = [1, 2, 3, 4, 2, 1, 0, 5, 3, 3, 1, 2];
  const a = new PcaModel(1);
  a.fit(new Float64Array(data), 2);
  const b = new PcaModel(1);
  b.fitF32(new Float32Array(data), 2);
  const ca = Array.from(a.components);
  const cb = Array.from(b.components);
  const sign = ca[0] * cb[0] + ca[1] * cb[1] < 0 ? -1 : 1;
  for (let i = 0; i < ca.length; i++) {
    assert.ok(Math.abs(ca[i] - sign * cb[i]) < 1e-4);
  }
});

test("PCA before fit throws", () => {
  assert.throws(() => new PcaModel(1).transform(new Float64Array(4), 2), /not fitted/);
});
