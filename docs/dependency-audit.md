# Dependency licence audit

Audit of the dependency graphs of the sibling TPT repositories that tpt-ml-proven
builds on (`tpt-math-*`, `tpt-formal`, `tpt-telos`), checked for Apache-2.0-only
licences. Method: `cargo metadata --locked` in each repo, then a scan of every
package's `license` field, then `cargo tree --target all` to see which edges
reach each flagged crate.

Recorded on 2026-11-17.

| Repo | Packages | Apache-2.0-only (by `license` field) | In build graph? |
| --- | --- | --- | --- |
| tpt-math | 170 | `approx` 0.5.1, `ciborium` / `ciborium-io` / `ciborium-ll` 0.2.2, `borsh-derive` 1.8.1 | `approx`: dev-dependency only. `ciborium*`: only via `criterion` (dev-dependency). `borsh-derive`: in the lockfile but not reached by any build edge. |
| tpt-formal | 20 | none | — |
| tpt-telos | 110 | none | — |

Every other flagged-looking licence string (`Apache-2.0 OR MIT`,
`MIT/Apache-2.0`, `Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT`, …) is
dual-licensed and passes via its MIT option.

**Result:** no Apache-2.0-only crate is in a runtime build graph of the sibling
repos. Dev-only exceptions exist in tpt-math and are not inherited by tpt-ml-proven,
because `deny.toml` checks the whole graph for this project.

**Caveat:** this audit covers only the sibling repos' current lockfiles. Re-run it
when their dependencies change, and when the algorithm crates add dependencies of
their own (`cargo deny check` in CI is the ongoing gate).
