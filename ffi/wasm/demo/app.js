// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions
// The demo's logic, kept free of DOM and wasm loading so that tests/web-demo.test.mjs can run it
// under Node. index.html draws the result.

export const COLOURS = ["#0a84ff", "#ff9f0a", "#30d158", "#bf5af2", "#ff453a", "#64d2ff", "#ffd60a", "#8e8e93"];
export const MIN_K = 1;
export const MAX_K = 8;
export const DEFAULT_K = 3;
export const POINT_COUNT = 240;

// Three blobs in the unit square. `random` returns numbers in [0, 1); pass a seeded one for tests.
export function makePoints(count, random = Math.random) {
  const points = new Float64Array(count * 2);
  for (let i = 0; i < count; i++) {
    const centre = i % 3;
    points[2 * i] = 0.2 + 0.3 * centre + 0.12 * (random() - 0.5) * 2;
    points[2 * i + 1] = 0.3 + 0.2 * ((centre * 7) % 3) + 0.12 * (random() - 0.5) * 2;
  }
  return points;
}

// The input box can hold anything; keep it to a whole number in range.
export function clampK(value) {
  const k = Math.round(Number(value));
  if (!Number.isFinite(k)) return DEFAULT_K;
  return Math.max(MIN_K, Math.min(MAX_K, k));
}

// Fits KMeans with the given model class (the wasm KMeansModel, or a stand-in in tests).
// `centers` is flat, row-major, k rows of two columns.
export function fitClusters(KMeansModel, points, k, seed = 7) {
  const model = new KMeansModel({ nClusters: k, seed });
  model.fit(points, 2);
  const centers = [];
  const flat = model.centers;
  for (let c = 0; c < k; c++) centers.push([flat[2 * c], flat[2 * c + 1]]);
  return {
    k,
    labels: model.labels,
    centers,
    inertia: model.inertia,
    nIter: model.nIter,
  };
}

export function describeFit(result, nPoints) {
  return `${nPoints} points, ${result.k} clusters, inertia ${result.inertia.toFixed(3)}, ${result.nIter} iterations`;
}
