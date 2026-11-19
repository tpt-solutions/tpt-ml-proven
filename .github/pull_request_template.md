## What this changes

## Checks

- [ ] `just ci` passes, or the equivalent steps: fmt, clippy with `-D warnings`, tests, `cargo deny check`
- [ ] Every new Rust file has the SPDX header
- [ ] New or changed clauses in `docs/contracts/` have matching runtime checks (`tptc-verify` links them)
- [ ] A changed model field has a new format version, and the old golden files still load
- [ ] Docs and the relevant `todo.md` items are updated
