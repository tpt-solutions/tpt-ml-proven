// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! End-to-end tests of the `tpt-ml` binary: fit, predict, transform, info and verify, run as
//! separate processes on real files.

use std::path::PathBuf;
use std::process::{Command, Output};

fn tpt_ml(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tpt-ml"))
        .args(args)
        .output()
        .expect("the tpt-ml binary runs")
}

/// A scratch directory unique to this test, under cargo's test target directory.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("cli-{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write(dir: &std::path::Path, name: &str, text: &str) -> String {
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn path(dir: &std::path::Path, name: &str) -> String {
    dir.join(name).to_string_lossy().into_owned()
}

/// Two well-separated blobs of three points each, with a header line.
const BLOBS: &str = "x,y\n0.0,0.0\n0.1,0.0\n0.0,0.1\n10.0,10.0\n10.1,10.0\n10.0,10.1\n";

/// y = 2·x0 − x1 + 3, exactly.
const LINEAR_X: &str = "0,0\n1,0\n0,1\n1,1\n2,3\n3,1\n";
const LINEAR_Y: &str = "3\n5\n2\n4\n4\n8\n";

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

#[test]
fn kmeans_fit_predict_and_verify() {
    let dir = scratch("kmeans");
    let input = write(&dir, "blobs.csv", BLOBS);
    let model = path(&dir, "kmeans.tpml");

    let fit = tpt_ml(&[
        "fit", "kmeans", "--input", &input, "--output", &model, "--k", "2",
    ]);
    assert!(fit.status.success(), "{}", stderr(&fit));

    let predict = tpt_ml(&["predict", &model, "--input", &input]);
    assert!(predict.status.success(), "{}", stderr(&predict));
    let text = stdout(&predict);
    let labels: Vec<&str> = text.lines().collect();
    assert_eq!(labels.len(), 6);
    // The first three points share a cluster, and the last three share the other one.
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[3], labels[4]);
    assert_ne!(labels[0], labels[3]);

    let verify = tpt_ml(&["verify", &model, "--data", &input]);
    assert!(
        verify.status.success(),
        "{}{}",
        stdout(&verify),
        stderr(&verify)
    );
    let report = stdout(&verify);
    assert!(report.contains("verified:"), "{report}");
    assert!(!report.contains("FAIL"), "{report}");
}

#[test]
fn fitting_twice_with_the_same_seed_gives_the_same_file() {
    let dir = scratch("determinism");
    let input = write(&dir, "blobs.csv", BLOBS);
    let a = path(&dir, "a.tpml");
    let b = path(&dir, "b.tpml");
    for out in [&a, &b] {
        let fit = tpt_ml(&[
            "fit", "kmeans", "--input", &input, "--output", out, "--k", "2", "--seed", "5",
        ]);
        assert!(fit.status.success(), "{}", stderr(&fit));
    }
    assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
}

#[test]
fn ridge_needs_a_target_and_then_predicts_it() {
    let dir = scratch("ridge");
    let input = write(&dir, "x.csv", LINEAR_X);
    let target = write(&dir, "y.csv", LINEAR_Y);
    let model = path(&dir, "ridge.tpml");

    let missing = tpt_ml(&["fit", "ridge", "--input", &input, "--output", &model]);
    assert_eq!(missing.status.code(), Some(1));
    assert!(
        stderr(&missing).contains("--target"),
        "{}",
        stderr(&missing)
    );

    let fit = tpt_ml(&[
        "fit",
        "ridge",
        "--input",
        &input,
        "--target",
        &target,
        "--output",
        &model,
        "--alpha",
        "0.000000001",
    ]);
    assert!(fit.status.success(), "{}", stderr(&fit));

    let predict = tpt_ml(&["predict", &model, "--input", &input]);
    let values: Vec<f64> = stdout(&predict)
        .lines()
        .map(|l| l.parse().unwrap())
        .collect();
    for (got, want) in values.iter().zip(LINEAR_Y.lines()) {
        let want: f64 = want.parse().unwrap();
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }
}

#[test]
fn pca_transform_has_one_column_per_component() {
    let dir = scratch("pca");
    let input = write(&dir, "x.csv", LINEAR_X);
    let model = path(&dir, "pca.tpml");
    let fit = tpt_ml(&[
        "fit",
        "pca",
        "--input",
        &input,
        "--output",
        &model,
        "--components",
        "1",
    ]);
    assert!(fit.status.success(), "{}", stderr(&fit));

    let transform = tpt_ml(&["transform", &model, "--input", &input]);
    assert!(transform.status.success(), "{}", stderr(&transform));
    let text = stdout(&transform);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 6);
    assert!(lines.iter().all(|l| l.split(',').count() == 1));
}

#[test]
fn info_names_the_kind_and_format() {
    let dir = scratch("info");
    let input = write(&dir, "x.csv", LINEAR_X);
    let target = write(&dir, "y.csv", LINEAR_Y);
    let model = path(&dir, "lasso.tpml");
    let fit = tpt_ml(&[
        "fit", "lasso", "--input", &input, "--target", &target, "--output", &model,
    ]);
    assert!(fit.status.success(), "{}", stderr(&fit));

    let info = tpt_ml(&["info", &model]);
    assert!(info.status.success(), "{}", stderr(&info));
    let text = stdout(&info);
    assert!(text.contains("kind:           lasso"), "{text}");
    assert!(text.contains("format:         version 2"), "{text}");
    assert!(text.contains("dtype:          f64"), "{text}");
}

#[test]
fn verify_rejects_a_corrupted_byte() {
    let dir = scratch("corrupt");
    let input = write(&dir, "blobs.csv", BLOBS);
    let model = path(&dir, "good.tpml");
    let fit = tpt_ml(&[
        "fit", "kmeans", "--input", &input, "--output", &model, "--k", "2",
    ]);
    assert!(fit.status.success(), "{}", stderr(&fit));

    let mut bytes = std::fs::read(&model).unwrap();
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0x01;
    let bad = path(&dir, "bad.tpml");
    std::fs::write(&bad, &bytes).unwrap();

    let verify = tpt_ml(&["verify", &bad, "--data", &input]);
    assert_eq!(verify.status.code(), Some(1), "{}", stdout(&verify));
    let report = stdout(&verify);
    assert!(report.contains("FAIL  checksum"), "{report}");
    assert!(report.contains("NOT verified"), "{report}");
}

#[test]
fn a_file_that_is_not_a_model_is_refused() {
    let dir = scratch("notmodel");
    let junk = write(
        &dir,
        "junk.tpml",
        "this is not a model file at all, just text",
    );
    let verify = tpt_ml(&["verify", &junk]);
    assert_eq!(verify.status.code(), Some(1));
    assert!(
        stdout(&verify).contains("FAIL  magic"),
        "{}",
        stdout(&verify)
    );
}

#[test]
fn bad_usage_exits_with_status_two() {
    let unknown = tpt_ml(&[
        "fit",
        "not-an-estimator",
        "--input",
        "x.csv",
        "--output",
        "m.tpml",
    ]);
    assert_eq!(unknown.status.code(), Some(2));
    let no_command = tpt_ml(&["bogus"]);
    assert_eq!(no_command.status.code(), Some(2));
    let help = tpt_ml(&["--help"]);
    assert!(help.status.success());
    assert!(stdout(&help).contains("ESTIMATORS"));
}

#[test]
fn predict_on_a_transformer_is_a_clear_error() {
    let dir = scratch("predict-transformer");
    let input = write(&dir, "x.csv", LINEAR_X);
    let model = path(&dir, "scaler.tpml");
    let fit = tpt_ml(&[
        "fit",
        "standard-scaler",
        "--input",
        &input,
        "--output",
        &model,
    ]);
    assert!(fit.status.success(), "{}", stderr(&fit));
    let predict = tpt_ml(&["predict", &model, "--input", &input]);
    assert_eq!(predict.status.code(), Some(1));
    assert!(
        stderr(&predict).contains("use transform"),
        "{}",
        stderr(&predict)
    );
}
