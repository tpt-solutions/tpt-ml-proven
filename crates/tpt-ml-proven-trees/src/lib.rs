// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Tree models for tpt-ml-proven.
//!
//! * [`DecisionTreeRegressor`] is one CART tree with exact greedy splits.
//! * [`GradientBoostingRegressor`] adds shrunken trees to a mean prediction, fitting each
//!   tree to the current residuals with squared-error loss. It matches scikit-learn's
//!   `GradientBoostingRegressor(loss="squared_error", subsample=1.0, max_features=None)`
//!   on committed fixtures.
//!
//! The structural invariants are contracts (see `docs/contracts/decision-tree.tptc` and
//! `docs/contracts/gradient-boosting.tptc`):
//!
//! * the leaves of every tree partition the training rows;
//! * every tree respects `max_depth`;
//! * each boosting round does not increase the training squared error.
//!
//! The last one holds because a leaf moves to the mean residual of its rows, and a step of
//! size `learning_rate ∈ (0, 1]` towards that optimum cannot increase a convex quadratic.
//!
//! Computation is in `f64`. Trees are built in O(n · cols · log n) per level.

mod gbm;
mod persist;
mod tree;

pub use gbm::{DecisionTreeRegressor, GradientBoostingRegressor};
pub use tree::Tree;
