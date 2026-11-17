tpt-ml-proven — Design Specification
Status: Draft
Maintainer: TPT Solutions
License: MIT OR Apache-2.0
Home: github.com/tpt-solutions/tpt-ml-proven
1. Problem Statement
Python's scikit-learn is the de-facto classical ML library, but it carries 2007-era API decisions, C/C++ extension fragility, the GIL, and zero correctness guarantees beyond unit tests. The Rust ecosystem's best answer (linfa) is incomplete — missing sparse data support, serialization, and many unsupervised techniques.
scikit-learn is also widely used as an FFI target. Wrapping it from Go, Java, or Rust is painful because its C ABI is non-existent.
2. Goals
Proven correctness. Core algorithms carry requires/ensures contracts verified by a self-contained QF_LRA solver. Not "it passed the tests" — "the math guarantees it."
Modern API. Type-safe estimators, zero-copy where possible, no_std-friendly cores, no GIL, true parallelism.
Scikit-compatible surface. Familiar .fit() / .transform() / .predict() semantics so Python users can migrate with minimal friction — but implemented cleanly, not aped.
FFI-first. Python (PyO3) and Go (cgo) bindings generated automatically, not hand-rolled.
Ecosystem-native. Built on tpt-math for numerics. No duplicated linear algebra.
No Apache only dependencies
3. Non-Goals
Deep learning / autograd. That is tpt-cobalt / tpt-ml's territory.
Wrapping scikit-learn. We are a replacement, not a bridge.
Being a general-purpose statistics library. That is tpt-math-prob / tpt-math-stats.
Competing with polars / pandas on dataframes. We consume tabular data; we do not own it.
4. Architecture

┌──────────────────────────────────────────────────────────┐
│  FFI surface  (PyO3 · cgo · WASM)                        │
├──────────────────────────────────────────────────────────┤
│  tpt-ml-proven  (high-level estimators)                  │
│   ├─ clustering   (KMeans, DBSCAN, HDBSCAN)              │
│   ├─ decomposition (PCA, TruncatedSVD, NMF)              │
│   ├─ linear_model  (Ridge, Lasso, ElasticNet, Logistic)  │
│   ├─ ensemble      (stump forests, gradient boosting)    │
│   └─ preprocessing (scalers, encoders, imputers)         │
├──────────────────────────────────────────────────────────┤
│  tpt-ml-proven-core  (traits, contracts, solver glue)    │
├──────────────────────────────────────────────────────────┤
│  tpt-math  (linalg, optimize, prob, signal)              │
└──────────────────────────────────────────────────────────┘

tpt-ml-proven-core defines the estimator traits and the contract language. tpt-ml-proven contains the algorithm implementations. Both depend only on tpt-math leaves.
5. Core Abstractions
The API is modeled on the estimator pattern but cleaned up with Rust's type system.

/// A supervised estimator: learns from (X, y) pairs.
pub trait SupervisedEstimator {
    type X: ArrayView;
    type Y: TargetView;
    type FitError: Debug;

    fn fit(&mut self, x: Self::X, y: Self::Y) -> Result<(), Self::FitError>;
}

/// An unsupervised estimator: learns from X alone.
pub trait UnsupervisedEstimator {
    type X: ArrayView;
    type FitError: Debug;

    fn fit(&mut self, x: Self::X) -> Result<(), Self::FitError>;
}

/// Anything that can produce predictions.
pub trait Predictor {
    type X: ArrayView;
    type Output;

    fn predict(&self, x: Self::X) -> Self::Output;
}

/// Anything that can transform data (preprocessing, decomposition).
pub trait Transformer {
    type X: ArrayView;
    type Output;

    fn transform(&self, x: Self::X) -> Self::Output;
    fn fit_transform(&mut self, x: Self::X) -> Self::Output;
}

Key differences from scikit:
fit returns Result, not silent mutation.
ArrayView is a zero-copy trait over tpt-math-linalg-dense and tpt-math-linalg-sparse — no implicit numpy copies.
Composition uses Rust generics and impl blocks, not a runtime Pipeline object.
6. Verification Model
Every core algorithm carries contracts expressed in the tpt-formal contract language and checked by the QF_LRA solver.
Example — K-Means:

requires: X.rows >= 1
requires: k >= 1
requires: k <= X.rows
ensures:  inertia(state_t+1) <= inertia(state_t)        // monotonic descent
ensures:  converged => forall i. assignments_t[i] == assignments_t+1[i]
ensures:  forall c in centers. c in convex_hull(X)

Example — PCA:

requires: X.rows >= 2
requires: 1 <= n_components <= min(X.rows, X.cols)
ensures:  forall i != j. dot(component_i, component_j) == 0    // orthogonality
ensures:  forall i. ||component_i|| == 1                        // unit norm
ensures:  sum(explained_variance) <= total_variance

Contracts are checked at compile time where possible (static shapes, constant k), and at runtime with cheap assertions where they cannot be (data-dependent bounds). The solver is self-contained — no external Z3 dependency.
7. Module Layout

tpt-ml-proven/
├── Cargo.toml                       # workspace
├── crates/
│   ├── tpt-ml-proven-core/          # traits, contracts, errors
│   ├── tpt-ml-proven-clustering/    # KMeans, DBSCAN, HDBSCAN
│   ├── tpt-ml-proven-decomposition/ # PCA, TruncatedSVD, NMF
│   ├── tpt-ml-proven-linear/        # Ridge, Lasso, ElasticNet, Logistic
│   ├── tpt-ml-proven-preprocessing/ # scalers, encoders, imputers
│   └── tpt-ml-proven-ffi/           # PyO3 + cgo glue
├── ffi/
│   ├── python/                      # maturin-built wheel
│   └── go/                          # cgo package
└── docs/
    ├── DESIGN.md                    # this file
    └── contracts/                   # one .tptc file per algorithm

Each algorithm crate depends only on tpt-ml-proven-core and the specific tpt-math-* leaves it needs.
8. Wedge Deliverables (v0.1)
Ship two algorithms end-to-end to prove the stack works:
KMeans — clustering. Demonstrates monotonic-convergence contract, sparse/dense input, multi-threaded Lloyd iteration.
PCA — decomposition. Demonstrates orthogonality contract, tight integration with tpt-math-linalg-dense eigensolvers.
Both ship with:
Pure-Rust no_std core (alloc only).
Python wheel via PyO3/maturin.
Go package via cgo.
Contract proofs checked in CI.
9. FFI Surface
The FFI layer is intentionally thin and generated, not hand-written.
Python:

from tpt_ml_proven import KMeans
model = KMeans(n_clusters=8).fit(X)
labels = model.predict(X)

Go:

import "tptsolutions.co.nz/mlproven"
m := mlproven.NewKMeans(8)
m.Fit(X)
labels := m.Predict(X)

Both are generated from the same Rust API via tpt-telos's FFI bridge. The Python API mirrors scikit-learn's method names for migration ease; the Go API follows Go idioms.
10. Success Criteria
v0.1 is done when:
KMeans and PCA pass scikit-learn's numerical test vectors within 1e-9 tolerance.
All contracts in docs/contracts/ are solver-verified in CI.
Python wheel installs on Linux/macOS/Windows without a C compiler.
Go package builds without cgo header gymnastics.
Benchmark: KMeans.fit on 1M × 50 float32 is ≥ 2× faster than scikit-learn on the same hardware.
README explains the verification story in one paragraph.
11. Out of Scope (for now)
Tree-based models (Random Forest, GBM). These need a separate crate (tpt-ml-proven-trees) because their verification story is different (structural invariants, not numeric).
GPU backends. Defer until tpt-gpu is stable.
Serialization format. Use bincode / serde initially; design a stable on-disk format in v0.2.