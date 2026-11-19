// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions
// Behaviour of the WebAssembly bindings. Run with `node --test tests/` after
// `wasm-pack build --target nodejs --release --no-opt --out-dir pkg`.

import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const m = require("../pkg/tpt_ml_proven_wasm.js");

// Two blobs of three points each, row-major, 2 features.
const BLOBS = new Float64Array([0, 0, 0.1, 0, 0, 0.1, 10, 10, 10.1, 10, 10, 10.1]);
// y = 2*x0 - x1 + 3, exactly, six rows of two features.
const LINEAR_X = new Float64Array([0, 0, 1, 0, 0, 1, 1, 1, 2, 3, 3, 1]);
const LINEAR_Y = Float64Array.from(Array.from({ length: 6 }, (_, r) => 2 * LINEAR_X[2 * r] - LINEAR_X[2 * r + 1] + 3));

function canonical(labels) {
  const seen = [];
  return Array.from(labels, (l) => {
    if (!seen.includes(l)) seen.push(l);
    return seen.indexOf(l);
  });
}

function notFitted(model, call) {
  assert.throws(call, /not fitted/, `${model} should refuse before fit`);
}

// ---------------------------------------------------------------- options

test("the version is exposed", () => {
  assert.match(m.version(), /^\d+\.\d+\.\d+/);
});

test("options must be an object", () => {
  assert.throws(() => new m.KMeansModel("three"), /options must be an object/);
});

test("a required option is named when it is missing", () => {
  assert.throws(() => new m.KMeansModel({}), /option nClusters is required/);
});

test("a seed must be a number, not a BigInt", () => {
  assert.throws(() => new m.KMeansModel({ nClusters: 2, seed: 1n }), /option seed must be a number/);
});

test("a count must be a non-negative integer", () => {
  assert.throws(() => new m.KMeansModel({ nClusters: 2.5 }), /non-negative integer/);
  assert.throws(() => new m.KMeansModel({ nClusters: -1 }), /non-negative integer/);
});

// ---------------------------------------------------------------- clustering

test("KMeans recovers blobs in f64 and f32", () => {
  const a = new m.KMeansModel({ nClusters: 2, seed: 1 });
  a.fit(BLOBS, 2);
  assert.deepEqual(canonical(a.labels), [0, 0, 0, 1, 1, 1]);
  assert.ok(a.inertia < 0.1);
  assert.equal(a.centers.length, 4);

  const b = new m.KMeansModel({ nClusters: 2, seed: 1 });
  b.fitF32(new Float32Array(BLOBS), 2);
  assert.deepEqual(canonical(b.labels), canonical(a.labels));
});

test("KMeans predict and transform have the documented shapes", () => {
  const km = new m.KMeansModel({ nClusters: 2, init: "random", seed: 1 });
  km.fit(BLOBS, 2);
  assert.deepEqual(Array.from(km.predict(BLOBS, 2)), Array.from(km.labels));
  const d = km.transform(BLOBS, 2);
  assert.equal(d.length, 6 * 2);
  assert.ok(d.every((v) => v >= 0));
});

test("KMeans rejects an unknown init", () => {
  assert.throws(() => new m.KMeansModel({ nClusters: 2, init: "bogus" }), /init must be/);
});

test("KMeans refuses to predict before fit", () => {
  notFitted("KMeansModel", () => new m.KMeansModel({ nClusters: 2 }).predict(BLOBS, 2));
});

test("DBSCAN labels the blobs and marks noise", () => {
  const db = new m.DbscanModel({ eps: 0.5, minSamples: 2 });
  db.fit(new Float64Array([...BLOBS, 50, 50]), 2);
  assert.deepEqual(Array.from(db.labels).slice(0, 6), [0, 0, 0, 1, 1, 1]);
  assert.equal(db.labels[6], -1);
  assert.ok(db.coreSampleIndices.length > 0);
});

test("HDBSCAN returns labels for the blobs", () => {
  const hd = new m.HdbscanModel({ minClusterSize: 3 });
  hd.fit(BLOBS, 2);
  assert.equal(hd.labels.length, 6);
});

// ---------------------------------------------------------------- decomposition

test("PCA transforms and inverts with the documented shapes", () => {
  const p = new m.PcaModel({ nComponents: 1 });
  p.fit(BLOBS, 2);
  const z = p.transform(BLOBS, 2);
  assert.equal(z.length, 6);
  const back = p.inverseTransform(z, 1);
  assert.equal(back.length, BLOBS.length);
  assert.equal(p.components.length, 2);
  assert.equal(p.mean.length, 2);
});

test("PCA in f32 keeps the same shapes", () => {
  const p = new m.PcaModel({ nComponents: 1 });
  p.fitF32(new Float32Array(BLOBS), 2);
  assert.equal(p.transform(BLOBS, 2).length, 6);
});

test("truncated SVD reports its singular values", () => {
  const s = new m.TruncatedSvdModel({ nComponents: 2 });
  s.fit(BLOBS, 2);
  assert.equal(s.singularValues.length, 2);
  assert.ok(s.singularValues[0] >= s.singularValues[1]);
  assert.equal(s.transform(BLOBS, 2).length, 12);
});

test("NMF factorises non-negative data", () => {
  const nmf = new m.NmfModel({ nComponents: 1, seed: 2 });
  nmf.fit(BLOBS, 2);
  assert.equal(nmf.components.length, 2);
  assert.equal(nmf.transform(BLOBS, 2).length, 6);
});

// ---------------------------------------------------------------- linear models

test("Ridge recovers an exact linear relation", () => {
  const r = new m.RidgeModel({ alpha: 1e-9 });
  r.fit(LINEAR_X, 2, LINEAR_Y);
  assert.ok(Math.abs(r.coef[0] - 2) < 1e-6);
  assert.ok(Math.abs(r.coef[1] + 1) < 1e-6);
  assert.ok(Math.abs(r.intercept - 3) < 1e-6);
  assert.ok(r.score(LINEAR_X, 2, LINEAR_Y) > 0.999999);
});

test("Lasso and ElasticNet fit and predict", () => {
  const l = new m.LassoModel({ alpha: 0.01 });
  l.fit(LINEAR_X, 2, LINEAR_Y);
  assert.equal(l.predict(LINEAR_X, 2).length, 6);
  const e = new m.ElasticNetModel({ alpha: 0.01, l1Ratio: 0.5 });
  e.fit(LINEAR_X, 2, LINEAR_Y);
  assert.equal(e.coef.length, 2);
});

test("binary logistic regression keeps the original labels", () => {
  const x = new Float64Array([-2, 0, -1, 0, -0.5, 0, 0.5, 0, 1, 0, 2, 0]);
  const y = new Float64Array([3, 3, 3, 7, 7, 7]);
  const lr = new m.LogisticRegressionModel({ C: 10 });
  lr.fit(x, 2, y);
  assert.deepEqual(Array.from(lr.classes), [3, 7]);
  assert.deepEqual(Array.from(lr.predict(x, 2)), [3, 3, 3, 7, 7, 7]);
  assert.equal(lr.score(x, 2, y), 1);
  assert.equal(lr.predictProba(x, 2).length, 12);
});

test("multinomial logistic regression separates three groups", () => {
  const x = new Float64Array([0, 0, 0.1, 0, 5, 5, 5.1, 5, 0, 5, 0.1, 5]);
  const y = new Float64Array([0, 0, 1, 1, 2, 2]);
  const lr = new m.LogisticRegressionModel({ C: 5 });
  lr.fit(x, 2, y);
  assert.deepEqual(Array.from(lr.classes), [0, 1, 2]);
  assert.equal(lr.score(x, 2, y), 1);
});

// ---------------------------------------------------------------- ensembles

test("the stump forest learns a threshold rule", () => {
  const x = new Float64Array([0, 1, 1, 2, 2, 3, 3, 4, 10, 1, 11, 2, 12, 3, 13, 4]);
  const y = new Float64Array([0, 0, 0, 0, 1, 1, 1, 1]);
  const f = new m.StumpForestModel({ nEstimators: 15, seed: 4 });
  f.fit(x, 2, y);
  assert.equal(f.score(x, 2, y), 1);
});

// ---------------------------------------------------------------- preprocessing

test("the standard scaler centres and scales each feature", () => {
  const s = new m.StandardScalerModel();
  const out = s.fitTransform(BLOBS, 2);
  const col0 = Array.from({ length: 6 }, (_, r) => out[2 * r]);
  const mean = col0.reduce((a, b) => a + b, 0) / 6;
  assert.ok(Math.abs(mean) < 1e-12);
  assert.equal(s.outputCols, 2);
});

test("min-max and robust scalers transform to the documented ranges", () => {
  const mm = new m.MinMaxScalerModel();
  const out = mm.fitTransform(BLOBS, 2);
  assert.ok(Math.min(...out) >= 0 && Math.max(...out) <= 1 + 1e-12);
  const rb = new m.RobustScalerModel();
  assert.equal(rb.fitTransform(BLOBS, 2).length, BLOBS.length);
});

test("the one-hot encoder reports its output width", () => {
  const cats = new Float64Array([0, 1, 1, 2, 0, 0]);
  const e = new m.OneHotEncoderModel();
  e.fit(cats, 2);
  assert.equal(e.outputCols, 3 + 2);
  const out = e.transform(cats, 2);
  assert.equal(out.length, 3 * 5);
  assert.ok(out.every((v) => v === 0 || v === 1));
});

test("the ordinal encoder maps categories to their indices", () => {
  // Rows [5, 9] and [7, 9]: column 0 has categories {5, 7}, column 1 has {9}.
  const cats = new Float64Array([5, 9, 7, 9]);
  const e = new m.OrdinalEncoderModel();
  assert.deepEqual(Array.from(e.fitTransform(cats, 2)), [0, 0, 1, 0]);
});

test("the imputer fills NaN with the chosen statistic", () => {
  const holey = new Float64Array([1, NaN, 3, 4, 5, 6]);
  const im = new m.SimpleImputerModel({ strategy: "constant", fillValue: -9 });
  assert.deepEqual(Array.from(im.fitTransform(holey, 2)), [1, -9, 3, 4, 5, 6]);
  assert.throws(() => new m.SimpleImputerModel({ strategy: "nearest" }), /strategy must be/);
});

// ---------------------------------------------------------------- trees and forests

test("a regression tree fits a step exactly", () => {
  const x = new Float64Array([0, 1, 2, 3, 4, 5, 6, 7]);
  const y = new Float64Array([1, 1, 1, 1, 9, 9, 9, 9]);
  const t = new m.DecisionTreeRegressorModel({ maxDepth: 2 });
  t.fit(x, 1, y);
  assert.deepEqual(Array.from(t.predict(x, 1)), Array.from(y));
  assert.equal(t.score(x, 1, y), 1);
});

test("a classification tree reports its classes and accuracy", () => {
  const x = new Float64Array([0, 1, 2, 3, 4, 5, 6, 7]);
  const y = new Float64Array([0, 0, 0, 0, 1, 1, 1, 1]);
  const t = new m.DecisionTreeClassifierModel({ maxDepth: 2, criterion: "entropy" });
  t.fit(x, 1, y);
  assert.deepEqual(Array.from(t.classes), [0, 1]);
  assert.equal(t.score(x, 1, y), 1);
  assert.throws(() => new m.DecisionTreeClassifierModel({ criterion: "chaos" }), /criterion must be/);
});

test("gradient boosting and forests fit the signal", () => {
  const x = new Float64Array(Array.from({ length: 40 }, (_, i) => i / 4));
  const y = Float64Array.from(x, (v) => 2 * v + 1);
  const g = new m.GradientBoostingRegressorModel({ nEstimators: 40, learningRate: 0.3, maxDepth: 3, seed: 1 });
  g.fit(x, 1, y);
  assert.ok(g.score(x, 1, y) > 0.95);

  const rf = new m.RandomForestRegressorModel({ nEstimators: 20, maxDepth: 5, seed: 2 });
  rf.fit(x, 1, y);
  assert.ok(rf.score(x, 1, y) > 0.8);

  const labels = Float64Array.from(x, (v) => (v > 5 ? 1 : 0));
  const rc = new m.RandomForestClassifierModel({ nEstimators: 15, maxDepth: 4, seed: 3 });
  rc.fit(x, 1, labels);
  assert.ok(rc.score(x, 1, labels) > 0.9);
  assert.deepEqual(Array.from(rc.classes), [0, 1]);
});

test("a seed reproduces a random forest, given as a number", () => {
  const x = new Float64Array(Array.from({ length: 30 }, (_, i) => i));
  const y = Float64Array.from(x, (v) => (v % 3 === 0 ? 1 : 0));
  const fit = () => {
    const f = new m.RandomForestClassifierModel({ nEstimators: 9, maxDepth: 3, seed: 5 });
    f.fit(x, 1, y);
    return Array.from(f.predict(x, 1));
  };
  assert.deepEqual(fit(), fit());
});
