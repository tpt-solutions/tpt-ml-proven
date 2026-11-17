// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! End-to-end checks of the verifier against fixtures. The negative cases matter
//! as much as the positive ones: a verifier that proves everything would pass the
//! positive cases but fail these.

use std::path::{Path, PathBuf};

use tptc_verify::model::parse_value;
use tptc_verify::{check_file, Linkage, Status, VerifyError};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn check(name: &str) -> tptc_verify::FileReport {
    check_file(&fixtures().join(name), &fixtures()).expect("fixture readable")
}

fn status_of(report: &tptc_verify::FileReport, id: &str) -> Status {
    report
        .clauses
        .iter()
        .find(|c| c.id == id)
        .unwrap_or_else(|| panic!("no clause {id}"))
        .status
        .clone()
}

#[test]
fn provable_ensures_is_proven_and_requires_is_assumed() {
    let r = check("good.tptc");
    assert!(r.parse_errors.is_empty(), "{:?}", r.parse_errors);
    assert_eq!(status_of(&r, "x_positive"), Status::Assumed);
    assert_eq!(status_of(&r, "x_nonnegative"), Status::Proven);
    assert_eq!(status_of(&r, "x_at_least_one"), Status::Proven);
    assert!(!r.failed(true));
    assert_eq!(r.linkage, Some(Linkage::NotDeclared));
}

#[test]
fn false_ensures_is_disproved_with_a_counterexample() {
    let r = check("false_ensures.tptc");
    match status_of(&r, "k_at_least_rows") {
        Status::Disproved(model) => {
            let k = model.iter().find(|(n, _)| n == "k").expect("k in model");
            let rows = model
                .iter()
                .find(|(n, _)| n == "rows")
                .expect("rows in model");
            // The counterexample must satisfy the requires clause and violate the ensures.
            let k = parse_value(&k.1).unwrap_or_else(|| panic!("k = {}", k.1));
            let rows = parse_value(&rows.1).unwrap_or_else(|| panic!("rows = {}", rows.1));
            assert!(
                k <= rows,
                "counterexample violates requires: k={k} rows={rows}"
            );
            assert!(
                k < rows,
                "counterexample does not violate ensures: k={k} rows={rows}"
            );
        }
        other => panic!("expected Disproved, got {other:?}"),
    }
    assert!(r.failed(false), "a disproved clause must fail the run");
}

#[test]
fn nonlinear_clause_is_unverified_not_failed_by_default() {
    let r = check("nonlinear.tptc");
    assert!(matches!(status_of(&r, "product"), Status::Unverified(ref s) if s.contains("product")));
    assert!(
        matches!(status_of(&r, "division"), Status::Unverified(ref s) if s.contains("division"))
    );
    assert!(matches!(status_of(&r, "call"), Status::Unverified(ref s) if s.contains("inertia")));
    assert!(
        !r.failed(false),
        "unverified clauses must not fail by default"
    );
    assert!(r.failed(true), "--fail-on-unverified must fail them");
}

#[test]
fn contradictory_requires_are_reported_as_vacuous() {
    let r = check("vacuous.tptc");
    assert_eq!(r.verify_error, Some(VerifyError::VacuousRequires));
    assert!(r.failed(false));
}

#[test]
fn syntax_errors_are_reported_per_line_and_stop_the_run() {
    let r = check("parse_error.tptc");
    assert!(!r.parse_errors.is_empty());
    assert!(
        r.parse_errors.iter().any(|e| e.line == 4),
        "{:?}",
        r.parse_errors
    );
    assert!(r.clauses.is_empty());
    assert!(r.failed(false));
}

#[test]
fn clause_without_runtime_check_fails_linkage() {
    let r = check("linkage_missing.tptc");
    assert_eq!(r.linkage, Some(Linkage::Missing(vec!["no_check".into()])));
    assert!(r.failed(false));
}

#[test]
fn every_clause_with_a_runtime_check_passes_linkage() {
    let r = check("linkage_ok.tptc");
    assert_eq!(r.linkage, Some(Linkage::Ok));
    assert!(!r.failed(false));
}

#[test]
fn missing_source_file_fails_linkage() {
    let r = check("linkage_no_file.tptc");
    assert!(matches!(r.linkage, Some(Linkage::SourceMissing(_))));
    assert!(r.failed(false));
}

// Gate: every contract under docs/contracts/ must pass.
#[test]
fn repo_contracts_all_pass() {
    // The real contracts under docs/contracts/ must pass. This is the same gate
    // CI runs, so a contract that drifts from its implementation fails here too.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = root.join("docs/contracts");
    let files = tptc_verify::collect(&dir).expect("contracts dir");
    assert!(!files.is_empty());
    for f in files {
        let r = check_file(&f, &root).expect("readable");
        assert!(!r.failed(false), "{} failed: {r:?}", f.display());
    }
}
