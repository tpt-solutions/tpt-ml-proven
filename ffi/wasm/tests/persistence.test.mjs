// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions
// Model files from the WebAssembly bindings: toBytes writes a model file, and fromBytes restores
// a fitted model from one. Run with `node --test tests/persistence.test.mjs` after the nodejs
// build (node scripts/build.mjs, which writes pkg/).

import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const m = require("../pkg/tpt_ml_proven_wasm.js");

// Two blobs of three points each, row-major, 2 features.
const BLOBS = new Float64Array([0, 0, 0.1, 0, 0, 0.1, 10, 10, 10.1, 10, 10, 10.1]);
const BLOB_Y = new Float64Array([0, 0, 0, 1, 1, 1]);
// y = 2*x0 - x1 + 3, exactly, six rows of two features.
const LINEAR_X = new Float64Array([0, 0, 1, 0, 0, 1, 1, 1, 2, 3, 3, 1]);
const LINEAR_Y = Float64Array.from(
  Array.from({ length: 6 }, (_, r) => 2 * LINEAR_X[2 * r] - LINEAR_X[2 * r + 1] + 3),
);
// Three classes along one feature.
const THREE_X = new Float64Array([0, 0.1, 1, 1.1, 2, 2.1]);
const THREE_Y = new Float64Array([0, 0, 1, 1, 2, 2]);

function magicOf(bytes) {
  return Buffer.from(bytes.subarray(0, 8)).toString("latin1");
}

function asArray(values) {
  return Array.from(values);
}

// Fits a model, saves it, restores it, and checks that predictions agree.
function roundTrip(Model, fit, predict) {
  const original = new Model();
  fit(original);
  const bytes = original.toBytes();
  assert.ok(bytes instanceof Uint8Array, "toBytes returns a Uint8Array");
  assert.equal(magicOf(bytes), "TPTMLMDL");
  const restored = Model.fromBytes(bytes);
  assert.deepEqual(asArray(predict(restored)), asArray(predict(original)));
  return { original, restored, bytes };
}

// ---------------------------------------------------------------- clustering

test("KMeans round-trips in f64 and keeps its labels", () => {
  const a = new m.KMeansModel({ nClusters: 2, seed: 1 });
  a.fit(BLOBS, 2);
  const bytes = a.toBytes();
  const b = m.KMeansModel.fromBytes(bytes);
  assert.deepEqual(asArray(b.labels), asArray(a.labels));
  assert.deepEqual(asArray(b.predict(BLOBS, 2)), asArray(a.predict(BLOBS, 2)));
});

test("KMeans round-trips a single-precision model", () => {
  const a = new m.KMeansModel({ nClusters: 2, seed: 1 });
  a.fitF32(new Float32Array(BLOBS), 2);
  const b = m.KMeansModel.fromBytes(a.toBytes());
  assert.deepEqual(asArray(b.predict(BLOBS, 2)), asArray(a.predict(BLOBS, 2)));
});

test("toBytes refuses before fit", () => {
  assert.throws(() => new m.KMeansModel({ nClusters: 2 }).toBytes(), /not fitted/);
  assert.throws(() => new m.PcaModel({ nComponents: 1 }).toBytes(), /not fitted/);
});

test("fromBytes rejects bytes that are not a model file", () => {
  assert.throws(() => m.KMeansModel.fromBytes(new Uint8Array(32)), /not a tpt-ml-proven model file/);
  assert.throws(() => m.KMeansModel.fromBytes(new Uint8Array(4)), /not a tpt-ml-proven model file/);
});

test("fromBytes rejects a model file written by another estimator", () => {
  const ridge = new m.RidgeModel({ alpha: 0.1 });
  ridge.fit(LINEAR_X, 2, LINEAR_Y);
  assert.throws(() => m.KMeansModel.fromBytes(ridge.toBytes()));
});

test("PCA round-trips in f64 and f32", () => {
  const a = new m.PcaModel({ nComponents: 1 });
  a.fit(LINEAR_X, 2);
  const b = m.PcaModel.fromBytes(a.toBytes());
  assert.deepEqual(asArray(b.transform(LINEAR_X, 2)), asArray(a.transform(LINEAR_X, 2)));

  const c = new m.PcaModel({ nComponents: 1 });
  c.fitF32(new Float32Array(LINEAR_X), 2);
  const d = m.PcaModel.fromBytes(c.toBytes());
  assert.deepEqual(asArray(d.transform(LINEAR_X, 2)), asArray(c.transform(LINEAR_X, 2)));
});

test("TruncatedSVD and NMF round-trip", () => {
  const svd = new m.TruncatedSvdModel({ nComponents: 1 });
  svd.fit(LINEAR_X, 2);
  const s = m.TruncatedSvdModel.fromBytes(svd.toBytes());
  assert.deepEqual(asArray(s.transform(LINEAR_X, 2)), asArray(svd.transform(LINEAR_X, 2)));

  const nmf = new m.NmfModel({ nComponents: 1, seed: 1 });
  nmf.fit(LINEAR_X, 2);
  const n = m.NmfModel.fromBytes(nmf.toBytes());
  assert.deepEqual(asArray(n.transform(LINEAR_X, 2)), asArray(nmf.transform(LINEAR_X, 2)));
});

test("DBSCAN and HDBSCAN keep their labels", () => {
  const d = new m.DbscanModel({ eps: 0.5, minSamples: 2 });
  d.fit(BLOBS, 2);
  assert.deepEqual(asArray(m.DbscanModel.fromBytes(d.toBytes()).labels), asArray(d.labels));

  const h = new m.HdbscanModel({ minClusterSize: 3 });
  h.fit(BLOBS, 2);
  assert.deepEqual(asArray(m.HdbscanModel.fromBytes(h.toBytes()).labels), asArray(h.labels));
});

// ---------------------------------------------------------------- linear

test("Ridge, Lasso and ElasticNet round-trip", () => {
  const ridge = new m.RidgeModel({ alpha: 0.1 });
  ridge.fit(LINEAR_X, 2, LINEAR_Y);
  assert.deepEqual(asArray(m.RidgeModel.fromBytes(ridge.toBytes()).predict(LINEAR_X, 2)), asArray(ridge.predict(LINEAR_X, 2)));

  const lasso = new m.LassoModel({ alpha: 0.01 });
  lasso.fit(LINEAR_X, 2, LINEAR_Y);
  assert.deepEqual(asArray(m.LassoModel.fromBytes(lasso.toBytes()).predict(LINEAR_X, 2)), asArray(lasso.predict(LINEAR_X, 2)));

  const net = new m.ElasticNetModel({ alpha: 0.01, l1Ratio: 0.5 });
  net.fit(LINEAR_X, 2, LINEAR_Y);
  assert.deepEqual(asArray(m.ElasticNetModel.fromBytes(net.toBytes()).predict(LINEAR_X, 2)), asArray(net.predict(LINEAR_X, 2)));
});

test("binary logistic regression round-trips with its probabilities", () => {
  const a = new m.LogisticRegressionModel({});
  a.fit(LINEAR_X, 2, BLOB_Y);
  const b = m.LogisticRegressionModel.fromBytes(a.toBytes());
  assert.deepEqual(asArray(b.predict(LINEAR_X, 2)), asArray(a.predict(LINEAR_X, 2)));
  assert.deepEqual(asArray(b.predictProba(LINEAR_X, 2)), asArray(a.predictProba(LINEAR_X, 2)));
});

test("multiclass logistic regression round-trips", () => {
  const a = new m.LogisticRegressionModel({});
  a.fit(THREE_X, 1, THREE_Y);
  const b = m.LogisticRegressionModel.fromBytes(a.toBytes());
  assert.deepEqual(asArray(b.predict(THREE_X, 1)), asArray(a.predict(THREE_X, 1)));
  assert.deepEqual(asArray(b.predictProba(THREE_X, 1)), asArray(a.predictProba(THREE_X, 1)));
});

// ---------------------------------------------------------------- ensemble and trees

test("StumpForest round-trips", () => {
  const a = new m.StumpForestModel({ nEstimators: 15, seed: 1 });
  a.fit(BLOBS, 2, BLOB_Y);
  const b = m.StumpForestModel.fromBytes(a.toBytes());
  assert.deepEqual(asArray(b.predict(BLOBS, 2)), asArray(a.predict(BLOBS, 2)));
});

test("the tree models round-trip", () => {
  roundTrip(
    m.DecisionTreeRegressorModel,
    (t) => t.fit(LINEAR_X, 2, LINEAR_Y),
    (t) => t.predict(LINEAR_X, 2),
  );
  roundTrip(
    m.DecisionTreeClassifierModel,
    (t) => t.fit(BLOBS, 2, BLOB_Y),
    (t) => t.predict(BLOBS, 2),
  );
  roundTrip(
    m.GradientBoostingRegressorModel,
    (t) => t.fit(LINEAR_X, 2, LINEAR_Y),
    (t) => t.predict(LINEAR_X, 2),
  );
  roundTrip(
    m.RandomForestRegressorModel,
    (t) => t.fit(LINEAR_X, 2, LINEAR_Y),
    (t) => t.predict(LINEAR_X, 2),
  );
  roundTrip(
    m.RandomForestClassifierModel,
    (t) => t.fit(BLOBS, 2, BLOB_Y),
    (t) => t.predict(BLOBS, 2),
  );
});

// ---------------------------------------------------------------- preprocessing

test("the scalers round-trip", () => {
  for (const Model of [m.StandardScalerModel, m.MinMaxScalerModel, m.RobustScalerModel]) {
    roundTrip(
      Model,
      (s) => s.fit(LINEAR_X, 2),
      (s) => s.transform(LINEAR_X, 2),
    );
  }
});

test("the encoders round-trip", () => {
  const cats = new Float64Array([0, 1, 1, 0, 2, 1, 0, 0]);
  roundTrip(
    m.OneHotEncoderModel,
    (e) => e.fit(cats, 2),
    (e) => e.transform(cats, 2),
  );
  roundTrip(
    m.OrdinalEncoderModel,
    (e) => e.fit(cats, 2),
    (e) => e.transform(cats, 2),
  );
});

test("the imputer keeps its strategy and fill value", () => {
  const withGaps = new Float64Array([1, NaN, 3, 4]);
  const a = new m.SimpleImputerModel({ strategy: "constant", fillValue: 7 });
  a.fit(withGaps, 2);
  const b = m.SimpleImputerModel.fromBytes(a.toBytes());
  assert.deepEqual(asArray(b.transform(withGaps, 2)), [1, 7, 3, 4]);
});
