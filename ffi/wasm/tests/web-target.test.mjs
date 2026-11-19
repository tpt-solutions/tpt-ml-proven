// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions
// The `web` build loaded as an ES module, the way a browser loads it, with the wasm bytes
// given to initSync (fetch is not available to Node for file URLs). Run after
// `node scripts/build.mjs`.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const webDir = join(here, "..", "pkg-web");
const web = await import(pathToFileURL(join(webDir, "tpt_ml_proven_wasm.js")).href);

test("the web build initialises from bytes and fits a model", () => {
  web.initSync({ module: readFileSync(join(webDir, "tpt_ml_proven_wasm_bg.wasm")) });
  const km = new web.KMeansModel({ nClusters: 2, seed: 1 });
  km.fit(new Float64Array([0, 0, 0.1, 0, 10, 10, 10.1, 10]), 2);
  assert.equal(km.labels.length, 4);
  assert.equal(km.labels[0], km.labels[1]);
  assert.notEqual(km.labels[0], km.labels[2]);
});
