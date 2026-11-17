// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Links each contract clause to a runtime check in the implementing source.
//!
//! The convention: clause `requires[id]` in contract `name` must appear in the
//! declared `source` file as the literal `"name.id"`, for example
//! `requires!(rows >= 1, "kmeans.rows_positive")`. This ties every `.tptc`
//! clause to code that enforces it, so a contract cannot drift from the
//! implementation without the check failing.

use std::path::Path;

use crate::parse::Contract;

/// Outcome of the linkage check for one contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Linkage {
    /// No `source` line. Linkage is skipped because the algorithm is not
    /// implemented yet.
    NotDeclared,
    /// Every clause has its runtime check.
    Ok,
    /// These clause ids have no matching check literal in the source.
    Missing(Vec<String>),
    /// The declared source file does not exist under the repo root.
    SourceMissing(String),
}

/// The literal a clause's runtime check must carry.
pub fn check_name(contract: &Contract, clause_id: &str) -> String {
    format!("{}.{}", contract.name, clause_id)
}

/// Checks linkage for `contract`, reading the source relative to `root`.
pub fn check(contract: &Contract, root: &Path) -> Linkage {
    let Some(source) = &contract.source else {
        return Linkage::NotDeclared;
    };
    let text = match std::fs::read_to_string(root.join(source)) {
        Ok(t) => t,
        Err(_) => return Linkage::SourceMissing(source.clone()),
    };
    let missing: Vec<String> = contract
        .clauses
        .iter()
        .filter(|c| !text.contains(&format!("\"{}\"", check_name(contract, &c.id))))
        .map(|c| c.id.clone())
        .collect();
    if missing.is_empty() {
        Linkage::Ok
    } else {
        Linkage::Missing(missing)
    }
}
