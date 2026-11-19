// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! The `tpt-ml` command: fit, predict and transform from CSV, inspect model files, and verify
//! them. [`run`] does the work, so tests can call it without starting a process.

pub mod args;
pub mod csv;
pub mod model;

use std::io::Write;

use tpt_ml_proven::persist::{FORMAT_VERSION, MAGIC};

use crate::args::Args;
use crate::model::{kind_name, Model, ESTIMATORS};

const USAGE: &str = "\
tpt-ml: fit, predict and transform with tpt-ml-proven models, from CSV files.

USAGE
  tpt-ml fit <estimator> --input X.csv --output MODEL [--target Y.csv] [options]
  tpt-ml predict <model> --input X.csv [--output FILE]
  tpt-ml transform <model> --input X.csv [--output FILE]
  tpt-ml info <model>
  tpt-ml verify <model> [--data X.csv]

CSV: comma-separated numbers, one sample per line. A first line that is not all numbers
is taken as a header. Targets are one value per line.

ESTIMATORS (with the defaults used when an option is not given)
  kmeans            --k 8 --seed 0
  pca               --components 2
  svd               --components 2
  nmf               --components 2 --seed 0
  standard-scaler
  minmax-scaler
  ridge             --alpha 1.0            (needs --target)
  lasso             --alpha 1.0            (needs --target)
  elasticnet        --alpha 1.0 --l1-ratio 0.5   (needs --target)
  logistic          --c 1.0                (needs --target, labels 0 and 1)

EXIT STATUS
  0 success, 1 a failure (including a failed verify), 2 bad usage
";

/// Runs one command. Returns the process exit status.
pub fn run(argv: &[String], out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    // Help is recognised before flag parsing, so `--help` never needs a value.
    if argv.is_empty() || argv.iter().any(|a| a == "--help" || a == "-h") {
        let _ = write!(out, "{USAGE}");
        return 0;
    }
    match dispatch(argv, out) {
        Ok(status) => status,
        Err(Failure::Usage(msg)) => {
            let _ = writeln!(err, "tpt-ml: {msg}\n\n{USAGE}");
            2
        }
        Err(Failure::Error(msg)) => {
            let _ = writeln!(err, "tpt-ml: {msg}");
            1
        }
    }
}

enum Failure {
    Usage(String),
    Error(String),
}

type Outcome = Result<i32, Failure>;

fn usage<T>(msg: impl Into<String>) -> Result<T, Failure> {
    Err(Failure::Usage(msg.into()))
}

fn dispatch(argv: &[String], out: &mut dyn Write) -> Outcome {
    let args = match Args::parse(argv) {
        Ok(a) => a,
        Err(msg) => return usage(msg),
    };
    if args.positional_len() == 0 {
        let _ = write!(out, "{USAGE}");
        return Ok(0);
    }
    let command = args.positional(0, "command").map_err(Failure::Usage)?;
    match command {
        "fit" => fit(&args, out),
        "predict" => predict(&args, out),
        "transform" => transform(&args, out),
        "info" => info(&args, out),
        "verify" => verify(&args, out),
        "help" => {
            let _ = write!(out, "{USAGE}");
            Ok(0)
        }
        other => usage(format!("unknown command '{other}'")),
    }
}

fn read_file(path: &str) -> Result<Vec<u8>, Failure> {
    std::fs::read(path).map_err(|e| Failure::Error(format!("cannot read {path}: {e}")))
}

fn read_text(path: &str) -> Result<String, Failure> {
    std::fs::read_to_string(path).map_err(|e| Failure::Error(format!("cannot read {path}: {e}")))
}

fn read_table(path: &str) -> Result<csv::Table, Failure> {
    csv::parse(&read_text(path)?).map_err(|e| Failure::Error(format!("{path}: {e}")))
}

/// Writes `text` to `--output` when given, otherwise to `out`.
fn emit(args: &Args, text: &str, out: &mut dyn Write) -> Result<(), Failure> {
    match args.flag("output") {
        Some(path) => std::fs::write(path, text)
            .map_err(|e| Failure::Error(format!("cannot write {path}: {e}"))),
        None => {
            let _ = out.write_all(text.as_bytes());
            Ok(())
        }
    }
}

fn check_flags(args: &Args, known: &[&str]) -> Result<(), Failure> {
    args.reject_unknown(known).map_err(Failure::Usage)
}

fn fit(args: &Args, out: &mut dyn Write) -> Outcome {
    check_flags(
        args,
        &[
            "input",
            "output",
            "target",
            "k",
            "components",
            "seed",
            "alpha",
            "l1-ratio",
            "c",
        ],
    )?;
    let name = args
        .positional(1, "estimator name")
        .map_err(Failure::Usage)?;
    if !ESTIMATORS.contains(&name) {
        return usage(format!("unknown estimator '{name}'"));
    }
    let table = read_table(args.required("input").map_err(Failure::Usage)?)?;
    let target = match args.flag("target") {
        Some(path) => {
            let text = read_text(path)?;
            Some(csv::parse_column(&text).map_err(|e| Failure::Error(format!("{path}: {e}")))?)
        }
        None => None,
    };
    let fitted = Model::fit(name, &table, target.as_deref(), args).map_err(Failure::Error)?;
    let output = args.required("output").map_err(Failure::Usage)?;
    let bytes = fitted.to_bytes().map_err(Failure::Error)?;
    std::fs::write(output, &bytes)
        .map_err(|e| Failure::Error(format!("cannot write {output}: {e}")))?;
    let _ = writeln!(
        out,
        "fitted {name} on {} rows and {} columns; wrote {output} ({} bytes)",
        table.rows,
        table.cols,
        bytes.len()
    );
    Ok(0)
}

fn predict(args: &Args, out: &mut dyn Write) -> Outcome {
    check_flags(args, &["input", "output"])?;
    let model_path = args.positional(1, "model file").map_err(Failure::Usage)?;
    let model = Model::from_bytes(&read_file(model_path)?).map_err(Failure::Error)?;
    let table = read_table(args.required("input").map_err(Failure::Usage)?)?;
    let values = model.predict(&table).map_err(Failure::Error)?;
    let mut text = String::new();
    for v in values {
        text.push_str(&v.to_string());
        text.push('\n');
    }
    emit(args, &text, out)?;
    Ok(0)
}

fn transform(args: &Args, out: &mut dyn Write) -> Outcome {
    check_flags(args, &["input", "output"])?;
    let model_path = args.positional(1, "model file").map_err(Failure::Usage)?;
    let model = Model::from_bytes(&read_file(model_path)?).map_err(Failure::Error)?;
    let table = read_table(args.required("input").map_err(Failure::Usage)?)?;
    let matrix = model.transform(&table).map_err(Failure::Error)?;
    let (rows, cols) = (
        tpt_ml_proven::ArrayView::rows(&matrix),
        tpt_ml_proven::ArrayView::cols(&matrix),
    );
    let text = csv::format_rows(rows, cols, matrix.as_slice());
    emit(args, &text, out)?;
    Ok(0)
}

/// Version and dtype from the header, read without loading the model.
fn header_summary(bytes: &[u8]) -> (u32, &'static str) {
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes"));
    let dtype = if version >= 2 && bytes.len() >= 20 {
        match u32::from_le_bytes(bytes[16..20].try_into().expect("4 bytes")) {
            1 => "f32",
            2 => "f64",
            _ => "unknown",
        }
    } else {
        "f64"
    };
    (version, dtype)
}

fn info(args: &Args, out: &mut dyn Write) -> Outcome {
    check_flags(args, &[])?;
    let path = args.positional(1, "model file").map_err(Failure::Usage)?;
    let bytes = read_file(path)?;
    let tag = model::file_kind(&bytes).map_err(Failure::Error)?;
    let (version, dtype) = header_summary(&bytes);
    let _ = writeln!(out, "file:           {path}");
    let _ = writeln!(
        out,
        "kind:           {}",
        kind_name(tag).unwrap_or("unknown")
    );
    let _ = writeln!(out, "format:         version {version}");
    let _ = writeln!(out, "dtype:          {dtype}");
    let _ = writeln!(out, "size:           {} bytes", bytes.len());
    match Model::from_bytes(&bytes) {
        Ok(_) => {
            let _ = writeln!(out, "status:         loads, checksum and invariants ok");
            Ok(0)
        }
        Err(msg) => {
            let _ = writeln!(out, "status:         {msg}");
            Ok(1)
        }
    }
}

/// One line of the verify report: a name, whether it passed, and what it found.
fn report(out: &mut dyn Write, name: &str, ok: bool, detail: &str) -> bool {
    let mark = if ok { "ok  " } else { "FAIL" };
    let _ = writeln!(out, "{mark}  {name:<18} {detail}");
    ok
}

fn verify(args: &Args, out: &mut dyn Write) -> Outcome {
    check_flags(args, &["data"])?;
    let path = args.positional(1, "model file").map_err(Failure::Usage)?;
    let bytes = read_file(path)?;
    let mut all_ok = true;

    let magic_ok = bytes.len() >= 8 && bytes[..8] == MAGIC;
    all_ok &= report(
        out,
        "magic",
        magic_ok,
        if magic_ok {
            "TPTMLMDL"
        } else {
            "not a tpt-ml-proven model file"
        },
    );

    let (version, dtype) = header_summary(&bytes);
    let version_ok = (1..=FORMAT_VERSION).contains(&version);
    all_ok &= report(
        out,
        "format version",
        version_ok,
        &format!("{version} (newest this build reads: {FORMAT_VERSION}), dtype {dtype}"),
    );

    let tag = model::file_kind(&bytes).ok();
    let kind_known = tag.and_then(kind_name).is_some();
    all_ok &= report(
        out,
        "model kind",
        kind_known,
        tag.and_then(kind_name).unwrap_or("unknown kind"),
    );

    // Loading checks the checksum, every field, and each model's invariants.
    let loaded = Model::from_bytes(&bytes);
    let load_ok = loaded.is_ok();
    let load_detail = match &loaded {
        Ok(m) => format!("checksum and invariants ok ({})", m.name()),
        Err(msg) => msg.clone(),
    };
    all_ok &= report(out, "checksum, fields", load_ok, &load_detail);

    if let Ok(model) = loaded {
        let round_trip = model.to_bytes();
        let same = matches!(&round_trip, Ok(b) if *b == bytes);
        all_ok &= report(
            out,
            "round trip",
            same,
            if same {
                "re-saving gives the same bytes"
            } else {
                "re-saving changed the bytes"
            },
        );

        match args.flag("data") {
            None => {
                let _ = writeln!(out, "skip  {:<18} no --data given", "runtime checks");
            }
            Some(data_path) => {
                let table = read_table(data_path)?;
                let result: Result<(usize, bool), String> = if model.predicts() {
                    model.predict(&table).map(|v| {
                        let finite = v.iter().all(|x| x.is_finite());
                        (v.len(), finite)
                    })
                } else {
                    model.transform(&table).map(|m| {
                        let finite = m.as_slice().iter().all(|x| x.is_finite());
                        (tpt_ml_proven::ArrayView::rows(&m), finite)
                    })
                };
                match result {
                    Ok((n, finite)) => {
                        all_ok &= report(
                            out,
                            "runtime checks",
                            finite,
                            &format!(
                                "{} on {n} rows, outputs {}",
                                if model.predicts() {
                                    "predict"
                                } else {
                                    "transform"
                                },
                                if finite {
                                    "all finite"
                                } else {
                                    "contain NaN or infinity"
                                }
                            ),
                        );
                    }
                    Err(msg) => {
                        all_ok &= report(out, "runtime checks", false, &msg);
                    }
                }
            }
        }
    } else {
        let _ = writeln!(out, "skip  {:<18} needs a file that loads", "round trip");
        let _ = writeln!(
            out,
            "skip  {:<18} needs a file that loads",
            "runtime checks"
        );
    }

    if all_ok {
        let _ = writeln!(out, "verified: {path}");
        Ok(0)
    } else {
        let _ = writeln!(out, "NOT verified: {path}");
        Ok(1)
    }
}
