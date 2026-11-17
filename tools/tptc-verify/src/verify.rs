// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Checks the clauses of a contract with `tpt-solver`.
//!
//! * A postcondition `G` is **proven** when every query of
//!   `hypotheses ∧ ¬G` (see [`smt::proof_queries`]) is UNSAT and the solver's
//!   checker accepts each answer. Strict relations are sent exactly.
//! * A postcondition is **disproved** only when a SAT model passes an exact
//!   check: every `requires` clause holds on it (leniently), and `G` is
//!   violated on it by a margin ([`model`]). A SAT answer that fails this check
//!   is reported as inconclusive, which would point to a solver bug.
//! * A `requires` clause that is not linear is left out of the solver's
//!   hypotheses. That only weakens the hypotheses, so proofs stay sound, and the
//!   model check still uses the full `requires` set.

use std::collections::HashMap;
use std::time::Duration;

use tpt_solver::api::{solve_str, Format, Model, Options, Report};
use tpt_solver::check::outcome::Outcome;
use tpt_solver::core::engine::SolveResult;

use crate::model;
use crate::parse::{Clause, Contract, Expr, Kind};
use crate::smt::{self, Cube};

/// Result for one clause.
#[derive(Debug, Clone, PartialEq)]
pub enum Status {
    /// A `requires` clause: assumed, not checked here. Linkage still requires
    /// its runtime check.
    Assumed,
    /// A postcondition the solver proved from the linear requires clauses.
    Proven,
    /// A postcondition refuted by a model that passed the exact check. The
    /// model is a counterexample.
    Disproved(Vec<(String, String)>),
    /// Outside QF_LRA, so the solver never saw it. The reason is attached.
    Unverified(String),
    /// The solver could not decide, or its answer could not be confirmed.
    Inconclusive(String),
}

/// A clause and its result.
#[derive(Debug, Clone, PartialEq)]
pub struct ClauseResult {
    /// Clause identifier.
    pub id: String,
    /// Requires or ensures.
    pub kind: Kind,
    /// What happened.
    pub status: Status,
}

/// Errors that stop verification of a whole contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    /// The solver's checker rejected its own answer. This is a solver bug.
    SolverRejected {
        /// Clause being checked when it happened.
        clause: String,
    },
    /// The requires clauses are contradictory, so every ensures would hold
    /// vacuously.
    VacuousRequires,
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerifyError::SolverRejected { clause } => write!(
                f,
                "solver checker rejected its own answer for clause `{clause}` (solver bug)"
            ),
            VerifyError::VacuousRequires => {
                write!(
                    f,
                    "requires clauses are contradictory; every ensures holds vacuously"
                )
            }
        }
    }
}

impl std::error::Error for VerifyError {}

/// Per-call solver budget. Each query is small.
const TIMEOUT: Duration = Duration::from_secs(60);

fn solve(script: &str) -> Result<Report, String> {
    let opts = Options {
        timeout: Some(TIMEOUT),
        ..Options::default()
    };
    solve_str(script, Format::SmtLib2, &opts).map_err(|e| e.to_string())
}

/// `true` if the solver proved the cube unsatisfiable and its checker accepted
/// that answer.
fn is_unsat(report: &Report) -> bool {
    report.claim == SolveResult::Unsat && report.verdict == Outcome::Accept
}

/// A model as the solver printed it, and the same values parsed to `f64`.
type ParsedModel = (Vec<(String, String)>, HashMap<String, f64>);

/// The solver's model as a name → value map, parsed to `f64`. `None` if there
/// is no model or a value cannot be parsed.
fn model_env(report: &Report) -> Option<ParsedModel> {
    let named = match &report.model {
        Some(Model::Named(v)) => v.clone(),
        _ => Vec::new(),
    };
    let mut env = HashMap::new();
    for (k, v) in &named {
        env.insert(k.clone(), model::parse_value(v)?);
    }
    Some((named, env))
}

/// Verifies every clause of `contract`.
pub fn verify(contract: &Contract) -> Result<Vec<ClauseResult>, VerifyError> {
    let requires: Vec<&Clause> = contract
        .clauses
        .iter()
        .filter(|c| c.kind == Kind::Requires)
        .collect();
    let linear_hyps: Vec<&Expr> = requires
        .iter()
        .filter(|c| smt::nonlinear_reason(&c.expr).is_none())
        .map(|c| &c.expr)
        .collect();
    let all_requires: Vec<&Expr> = requires.iter().map(|c| &c.expr).collect();

    // Vacuity: if every hypothesis cube is UNSAT (a sound check, because the
    // encoding is a superset), the requires clauses are contradictory.
    if !linear_hyps.is_empty() {
        if let Some(cubes) = smt::hypothesis_cubes(&linear_hyps) {
            let all_unsat = !cubes.is_empty()
                && cubes.iter().all(|cube| match solve(&smt::script(cube)) {
                    Ok(r) => is_unsat(&r),
                    Err(_) => false,
                });
            if all_unsat {
                return Err(VerifyError::VacuousRequires);
            }
        }
    }

    let mut out = Vec::with_capacity(contract.clauses.len());
    for clause in &contract.clauses {
        let status = match clause.kind {
            Kind::Requires => Status::Assumed,
            Kind::Ensures => check_ensures(clause, &linear_hyps, &all_requires)?,
        };
        out.push(ClauseResult {
            id: clause.id.clone(),
            kind: clause.kind,
            status,
        });
    }
    Ok(out)
}

fn check_ensures(
    clause: &Clause,
    linear_hyps: &[&Expr],
    all_requires: &[&Expr],
) -> Result<Status, VerifyError> {
    if let Some(reason) = smt::nonlinear_reason(&clause.expr) {
        return Ok(Status::Unverified(reason));
    }
    let Some(queries) = smt::proof_queries(linear_hyps, &clause.expr) else {
        return Ok(Status::Inconclusive(format!(
            "case split exceeds {} cubes",
            smt::MAX_CUBES
        )));
    };

    // Every query must be UNSAT for the ensures clause to hold. A SAT query is a
    // candidate counterexample, and it is reported only if the exact check confirms it.
    for cube in &queries {
        let Some(report) = run(clause, cube)? else {
            return Ok(Status::Inconclusive("solver did not answer".into()));
        };
        if is_unsat(&report) {
            continue;
        }
        if report.claim == SolveResult::Sat && report.verdict == Outcome::Accept {
            return Ok(confirm_counterexample(clause, all_requires, &report));
        }
        return Ok(Status::Inconclusive(format!(
            "solver answered {:?} with checker verdict {:?}",
            report.claim, report.verdict
        )));
    }
    Ok(Status::Proven)
}

/// Runs one query. `None` means the solver failed to answer (for example it does
/// not support the input); that is inconclusive, not a bug. A checker `Reject`
/// is a solver bug and stops verification.
fn run(clause: &Clause, cube: &Cube) -> Result<Option<Report>, VerifyError> {
    let Ok(report) = solve(&smt::script(cube)) else {
        return Ok(None);
    };
    if report.verdict == Outcome::Reject {
        return Err(VerifyError::SolverRejected {
            clause: clause.id.clone(),
        });
    }
    Ok(Some(report))
}

/// Accepts a SAT answer as a counterexample only if its model passes the exact
/// check against the original clauses.
fn confirm_counterexample(clause: &Clause, all_requires: &[&Expr], report: &Report) -> Status {
    let Some((named, env)) = model_env(report) else {
        return Status::Inconclusive("solver model has a value that cannot be parsed".into());
    };
    let requires_hold = all_requires
        .iter()
        .all(|r| model::holds_lenient(r, &env) == Some(true));
    let goal_violated = model::holds_robust(&model::negate(&clause.expr), &env) == Some(true);
    if requires_hold && goal_violated {
        Status::Disproved(named)
    } else {
        Status::Inconclusive(
            "solver counterexample does not satisfy the clauses (suspected solver bug)".into(),
        )
    }
}
