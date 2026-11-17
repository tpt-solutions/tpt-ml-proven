# Contracts

Each algorithm has one `.tptc` file here. The file states the algorithm's
`requires` (preconditions) and `ensures` (postconditions). The tool
`tools/tptc-verify` checks them in two ways:

1. **Solver check.** Linear clauses are checked with the QF_LRA solver from
   `tpt-solver`.
2. **Linkage.** Every clause must have a matching runtime check in the Rust
   source, so the contract cannot drift from the code.

Run it from the repo root:

```sh
cargo run --manifest-path tools/tptc-verify/Cargo.toml -- docs/contracts
```

Exit code 0 means no clause failed. `--fail-on-unverified` also fails on
clauses the solver cannot see (see [Verdicts](#verdicts)).

## Writing a contract

```text
# Comments start with '#'.
contract kmeans                   # required, once, first
source crates/.../src/lib.rs      # optional: file holding the runtime checks

requires[k_positive]: k >= 1
ensures[k_within_rows]: k <= X.rows
```

- `contract <name>`: the name prefixes every runtime check (see Linkage).
- `source <path>`: repo-relative Rust file that enforces the clauses. Omit it
  until the algorithm exists.
- `requires[id]:` / `ensures[id]:`: a clause. `id` is unique within the file.
- Expressions: numbers (`2`, `0.5`), identifiers (may contain `.`, so `X.rows`
  is one name), `+ - * /`, parentheses, relations `<= < >= > == !=`, and boolean
  `! && || =>`. A clause must be a comparison or a boolean combination of
  comparisons.
- Function application, e.g. `inertia(state)`, is parsed and treated as an
  uninterpreted function. Clauses that use one are unverified.
- Not supported: quantifiers (`forall`, `exists`). Rewrite them as a predicate
  call, as the KMeans and PCA contracts do, and check the predicate at runtime
  or with property tests.

## Verdicts

| Verdict | Meaning | Fails CI? |
| --- | --- | --- |
| `assumed` | A `requires` clause. It is a hypothesis, not checked. | no |
| `PROVEN` | The solver showed the clause follows from the linear requires clauses. | no |
| `DISPROVED` | A counterexample exists. The exact check confirmed it, and it is printed. | **yes** |
| `UNVERIFIED` | Not linear (product of variables, division by a variable, or a function call). Enforced by runtime checks and property tests. | only with `--fail-on-unverified` |
| `INCONCLUSIVE` | The solver could not decide, or its answer could not be confirmed. | **yes** |
| `PARSE ERROR` / `VERIFY ERROR` / `LINKAGE FAILED` | Malformed file, a solver checker rejection (a solver bug), contradictory requires clauses, or a clause without a runtime check. | **yes** |

Requires clauses that are not linear are excluded from the solver's
hypotheses. That only weakens them, so proofs stay sound. A counterexample must
still satisfy every requires clause, and the exact check enforces that.

## Linkage

For contract `name` and clause `id`, the source file must contain the string
literal `"name.id"` as the name of a runtime check:

```rust
requires!(k >= 1, "kmeans.k_positive");
ensures!(inertia_next <= inertia_prev, "kmeans.monotone_inertia");
```

`requires!` and `ensures!` are defined in `tpt-ml-proven-core`. The check
returns `Err(Error::Contract(..))` when it fails. The `runtime-contracts`
feature turns these checks off.

A contract with no `source` line gets the note "not implemented yet" and is not
linked. Linkage does not require a clause to be solver-proven. A proven clause
still needs its runtime check, which is the way to keep the code and the
contract in step.

## Solver limits (fixed)

Testing found two bugs in `tpt-solver`. Both are fixed on its `fix/strict-lra`
branch (merged to `master`, commit `b50487d`), and `tools/tptc-verify` is
pinned to that commit:

1. **Strict inequalities were approximated.** The SMT-LIB2 parser lowered `<`
   and `>` to `<=` and `>=`, so `x >= 1 && x < 1` came back SAT with `x = 1`.
   Strictness is now exact through the parser, Fourier–Motzkin, the simplex
   model search, and the checker.
2. **The checker accepted SAT models that violated the assertions.** The checker
   now rejects a model on a strict constraint when `lhs >= rhs`. The verifier
   also confirms every counterexample itself (`tools/tptc-verify/src/model.rs`)
   before reporting `DISPROVED`.

The solver does not accept disjunctions over arithmetic, so the verifier splits
formulas into conjunctions (DNF) and proves each one. Disequalities `a != b`
become `a < b || a > b`.
