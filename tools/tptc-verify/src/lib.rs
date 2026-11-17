// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! `tptc-verify`: checks `.tptc` contract files.
//!
//! For each file it parses the contract, solver-checks the linear clauses with
//! `tpt-solver`, reports nonlinear clauses as unverified, and checks that every
//! clause has a runtime check in the declared source file.

pub mod linkage;
pub mod model;
pub mod parse;
pub mod smt;
pub mod verify;

use std::path::{Path, PathBuf};

pub use linkage::Linkage;
pub use parse::{parse_contract, Contract, ParseError};
pub use verify::{ClauseResult, Status, VerifyError};

/// Everything learned about one `.tptc` file.
#[derive(Debug)]
pub struct FileReport {
    /// The file that was checked.
    pub path: PathBuf,
    /// Parse errors. When non-empty, nothing else ran.
    pub parse_errors: Vec<ParseError>,
    /// Stops verification (solver bug or vacuous requires).
    pub verify_error: Option<VerifyError>,
    /// Per-clause results. Empty if parsing or verification stopped.
    pub clauses: Vec<ClauseResult>,
    /// Linkage result, if verification ran.
    pub linkage: Option<Linkage>,
}

impl FileReport {
    /// `true` if this file is a hard failure: a parse error, a disproved or
    /// inconclusive clause, a solver error, or a linkage gap. Unverified
    /// (nonlinear) clauses count only when `fail_on_unverified` is set.
    pub fn failed(&self, fail_on_unverified: bool) -> bool {
        if !self.parse_errors.is_empty() || self.verify_error.is_some() {
            return true;
        }
        let clause_failure = self.clauses.iter().any(|c| match &c.status {
            Status::Disproved(_) | Status::Inconclusive(_) => true,
            Status::Unverified(_) => fail_on_unverified,
            _ => false,
        });
        let linkage_failure = matches!(
            self.linkage,
            Some(Linkage::Missing(_)) | Some(Linkage::SourceMissing(_))
        );
        clause_failure || linkage_failure
    }
}

/// Checks one `.tptc` file. `root` is the repo root that `source` paths are
/// relative to.
pub fn check_file(path: &Path, root: &Path) -> std::io::Result<FileReport> {
    let text = std::fs::read_to_string(path)?;
    let mut report = FileReport {
        path: path.to_path_buf(),
        parse_errors: Vec::new(),
        verify_error: None,
        clauses: Vec::new(),
        linkage: None,
    };
    let contract = match parse_contract(&text) {
        Ok(c) => c,
        Err(errs) => {
            report.parse_errors = errs;
            return Ok(report);
        }
    };
    match verify::verify(&contract) {
        Ok(clauses) => report.clauses = clauses,
        Err(e) => {
            report.verify_error = Some(e);
            return Ok(report);
        }
    }
    report.linkage = Some(linkage::check(&contract, root));
    Ok(report)
}

/// Lists the `.tptc` files in a directory (non-recursive, sorted), or returns
/// the path itself if it is a file.
pub fn collect(path: &Path) -> std::io::Result<Vec<PathBuf>> {
    if !path.is_dir() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "tptc"))
        .collect();
    files.sort();
    Ok(files)
}
