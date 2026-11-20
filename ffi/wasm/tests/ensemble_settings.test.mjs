// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions
// The forest and boosting settings in WebAssembly: the split size, the leaf limit and feature
// importances, and that a saved model keeps them. Run with
// `node --test tests/ensemble_settings.test.mjs` after the nodejs build.

import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const m = require("../pkg/tpt_ml_proven_wasm.js");

// Feature 0 decides three levels of y; feature 1 is constant.
const X = new Float64Array(Array.from({ length: 36 }, (_, i) => [i, 2]).flat());
const Y = Float64Array.from({ length: 36 }, (_, i) => (i < 12 ? 0 : i < 24 ? 5 : 9));
const LABELS = Float64Array.from(Y, (v) => v / 5);

test("a boosted model reports importances that sum to one and ignore the constant feature", () => {
  const g = new m.GradientBoostingRegressorModel({ nEstimators: 10, maxDepth: 3, maxLeafNodes: 4 });
  g.fit(X, 2, Y);
  const imp = Array.from(g.featureImportances);
  assert.ok(Math.abs(imp[0] + imp[1] - 1) < 1e-12);
  assert.equal(imp[1], 0);
});

test("a regression forest reports importances and keeps its settings through bytes", () => {
  const f = new m.RandomForestRegressorModel({ nEstimators: 8, maxDepth: 4, minSamplesSplit: 6, maxLeafNodes: 4, seed: 1 });
  f.fit(X, 2, Y);
  const back = m.RandomForestRegressorModel.fromBytes(f.toBytes());
  assert.deepEqual(Array.from(back.featureImportances), Array.from(f.featureImportances));
  assert.deepEqual(Array.from(back.predict(X, 2)), Array.from(f.predict(X, 2)));
});

test("a classification forest reports importances and keeps its settings through bytes", () => {
  const f = new m.RandomForestClassifierModel({ nEstimators: 8, maxDepth: 4, minSamplesSplit: 4, maxLeafNodes: 4, maxFeatures: 2, seed: 2 });
  f.fit(X, 2, LABELS);
  const back = m.RandomForestClassifierModel.fromBytes(f.toBytes());
  assert.deepEqual(Array.from(back.featureImportances), Array.from(f.featureImportances));
  assert.deepEqual(Array.from(back.predict(X, 2)), Array.from(f.predict(X, 2)));
});

test("a boosted model keeps its settings and importances through bytes", () => {
  const g = new m.GradientBoostingRegressorModel({ nEstimators: 6, maxDepth: 3, minSamplesSplit: 5, maxLeafNodes: 4 });
  g.fit(X, 2, Y);
  const back = m.GradientBoostingRegressorModel.fromBytes(g.toBytes());
  assert.deepEqual(Array.from(back.featureImportances), Array.from(g.featureImportances));
  assert.deepEqual(Array.from(back.predict(X, 2)), Array.from(g.predict(X, 2)));
});

test("importances are refused before fit", () => {
  assert.throws(() => new m.RandomForestRegressorModel({}).featureImportances, /not fitted/);
  assert.throws(() => new m.GradientBoostingRegressorModel({}).featureImportances, /not fitted/);
});
