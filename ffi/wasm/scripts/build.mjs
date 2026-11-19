// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions
// Builds the WebAssembly package for each target and writes its package.json from
// package.template.json. Run from ffi/wasm with `node scripts/build.mjs`. Needs wasm-pack.
//
//   pkg/        --target nodejs   (require() from Node)
//   pkg-web/    --target web      (ES module, loaded in a browser with init())
//   pkg-bundler/ --target bundler (ES module for webpack, Vite and similar)

import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");

const cargo = readFileSync(join(root, "Cargo.toml"), "utf8");
const version = cargo.match(/^version = "([^"]+)"/m)?.[1];
const description = cargo.match(/^description = "([^"]+)"/m)?.[1];
if (!version || !description) {
  throw new Error("Cargo.toml must give a version and a description");
}

const template = JSON.parse(readFileSync(join(root, "package.template.json"), "utf8"));

const targets = [
  { target: "nodejs", out: "pkg", main: "tpt_ml_proven_wasm.js", module: null, sideEffects: false },
  { target: "web", out: "pkg-web", main: "tpt_ml_proven_wasm.js", module: null, sideEffects: false },
  { target: "bundler", out: "pkg-bundler", main: "tpt_ml_proven_wasm.js", module: "tpt_ml_proven_wasm.js", sideEffects: ["./snippets/*"] },
];

for (const { target, out, main, module, sideEffects } of targets) {
  const build = spawnSync(
    "wasm-pack",
    ["build", "--target", target, "--release", "--no-opt", "--out-dir", out],
    { cwd: root, stdio: "inherit" },
  );
  if (build.status !== 0) {
    process.exit(build.status ?? 1);
  }

  const files = readdirSync(join(root, out)).filter((f) => f !== "package.json" && f !== ".gitignore");
  const pkg = JSON.parse(JSON.stringify(template)
    .replaceAll("@VERSION@", version)
    .replaceAll("@DESCRIPTION@", description)
    .replaceAll("\"@FILES@\"", JSON.stringify(files))
    .replaceAll("\"@SIDE_EFFECTS@\"", JSON.stringify(sideEffects))
    .replaceAll("@MAIN@", main)
    .replaceAll("@TYPES@", main.replace(/\.js$/, ".d.ts")));
  if (module) {
    pkg.module = module;
  } else {
    delete pkg.module;
  }
  writeFileSync(join(root, out, "package.json"), JSON.stringify(pkg, null, 2) + "\n");
  console.log(`wrote ${out}/package.json (${target})`);
}
