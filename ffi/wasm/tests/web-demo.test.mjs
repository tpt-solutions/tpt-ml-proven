// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions
// The browser demo's logic (demo/app.js) run against the web build, with the wasm bytes given to
// initSync. This does not draw anything: rendering in a real browser is checked by hand. Run after
// `node scripts/build.mjs`.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const webDir = join(here, "..", "pkg-web");
const web = await import(pathToFileURL(join(webDir, "tpt_ml_proven_wasm.js")).href);
const app = await import(pathToFileURL(join(here, "..", "demo", "app.js")).href);
web.initSync({ module: readFileSync(join(webDir, "tpt_ml_proven_wasm_bg.wasm")) });

// The same generator the page uses for its points, seeded so the test is repeatable.
function seeded(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// The three blob centres that makePoints draws around.
const TRUE_CENTRES = [
  [0.2, 0.3],
  [0.5, 0.5],
  [0.8, 0.7],
];

test("makePoints returns row-major pairs inside the unit square", () => {
  const points = app.makePoints(app.POINT_COUNT, seeded(1));
  assert.equal(points.length, app.POINT_COUNT * 2);
  for (const v of points) {
    assert.ok(v > -0.1 && v < 1.1, `coordinate ${v} outside the expected range`);
  }
});

test("clampK keeps the input in range and falls back for junk", () => {
  assert.equal(app.clampK(0), 1);
  assert.equal(app.clampK(99), 8);
  assert.equal(app.clampK(2.6), 3);
  assert.equal(app.clampK("4"), 4);
  assert.equal(app.clampK("abc"), app.DEFAULT_K);
  assert.equal(app.clampK(undefined), app.DEFAULT_K);
});

test("fitClusters on the demo data finds three centres near the blobs", () => {
  const points = app.makePoints(app.POINT_COUNT, seeded(2));
  const result = app.fitClusters(web.KMeansModel, points, 3);

  assert.equal(result.labels.length, app.POINT_COUNT);
  assert.equal(new Set(result.labels).size, 3);
  assert.equal(result.centers.length, 3);
  assert.ok(Number.isFinite(result.inertia) && result.inertia >= 0);
  assert.ok(result.nIter >= 1);

  for (const truth of TRUE_CENTRES) {
    const nearest = Math.min(...result.centers.map(([x, y]) => Math.hypot(x - truth[0], y - truth[1])));
    assert.ok(nearest < 0.05, `no fitted centre within 0.05 of [${truth}] (nearest ${nearest})`);
  }
});

test("fitClusters with one cluster puts every point in label 0", () => {
  const points = app.makePoints(app.POINT_COUNT, seeded(3));
  const result = app.fitClusters(web.KMeansModel, points, 1);
  assert.deepEqual([...new Set(result.labels)], [0]);
  assert.equal(result.centers.length, 1);
});

test("describeFit reports points, clusters, inertia and iterations", () => {
  const text = app.describeFit({ k: 3, inertia: 1.23456, nIter: 7 }, 240);
  assert.equal(text, "240 points, 3 clusters, inertia 1.235, 7 iterations");
});
