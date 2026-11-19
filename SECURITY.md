# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it privately through
GitHub's **private vulnerability reporting** on this repository (the "Report a vulnerability"
button under the Security tab). Include what you found, the version or commit, and a way to
reproduce it.

We aim to acknowledge a report within a week. Fixes are released as new versions, and the
advisory is published once a fix is available.

## Scope

In scope: memory safety and undefined behaviour in the Rust crates and the bindings, and
model files that cause a crash, an unbounded allocation or an incorrect load that passes
the checks in `docs/model-format.md`.

Out of scope: a model file that was deliberately re-checksummed after being altered. The
format's checksum catches corruption and truncation, not tampering. See the compatibility
notes in `docs/model-format.md`, and only load model files from sources you trust.
