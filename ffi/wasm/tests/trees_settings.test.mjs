// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions
// The single-tree settings in WebAssembly: the split size, the leaf limit and feature importances,
// and that a saved model keeps them. Run with `node --test tests/trees_settings.test.mjs` after the
// nodejs build.

import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const m = require("../pkg/tpt_ml_proven_wasm.js");

// Three levels of y against feature 0; feature 1 is constant.
const X = new Float64Array(Array.from({ length: 18 }, (_, i) => [i, 2]).flat());
const Y = Float64Array.from({ length: 18 }, (_, i) => (i < 6 ? 0 : i < 12 ? 5 : 9));

test("the leaf limit is respected", () => {
  const t = new m.DecisionTreeRegressorModel({ maxLeafNodes: 3 });
  t.fit(X, 2, Y);
  assert.deepEqual(Array.from(t.predict(X, 2)), Array.from(Y));
});

test("feature importances are shares that sum to one, and the constant feature gets none", () => {
  const t = new m.DecisionTreeRegressorModel({});
  t.fit(X, 2, Y);
  const imp = Array.from(t.featureImportances);
  assert.equal(imp.length, 2);
  assert.ok(Math.abs(imp[0] + imp[1] - 1) < 1e-12);
  assert.equal(imp[1], 0);
});

test("importances are refused before fit", () => {
  assert.throws(() => new m.DecisionTreeClassifierModel({}).featureImportances, /not fitted/);
});

test("a classifier keeps its settings and importances through bytes", () => {
  const labels = Float64Array.from({ length: 18 }, (_, i) => (i < 9 ? 0 : 1));
  const a = new m.DecisionTreeClassifierModel({ minSamplesSplit: 4, maxLeafNodes: 4 });
  a.fit(X, 2, labels);
  const b = m.DecisionTreeClassifierModel.fromBytes(a.toBytes());
  assert.deepEqual(Array.from(b.predict(X, 2)), Array.from(a.predict(X, 2)));
  assert.deepEqual(Array.from(b.featureImportances), Array.from(a.featureImportances));
});

test("a regressor keeps its split size and leaf limit through bytes", () => {
  const a = new m.DecisionTreeRegressorModel({ minSamplesSplit: 19 });
  a.fit(X, 2, Y);
  // With 18 rows and a split size of 19 the root cannot split, so the model is one leaf.
  const b = m.DecisionTreeRegressorModel.fromBytes(a.toBytes());
  assert.deepEqual(Array.from(b.predict(X, 2)), Array.from(a.predict(X, 2)));
  assert.equal(new Set(Array.from(b.predict(X, 2))).size, 1);
});
