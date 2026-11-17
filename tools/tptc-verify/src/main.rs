// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

use std::path::PathBuf;
use std::process::ExitCode;

use tptc_verify::parse::Kind;
use tptc_verify::{check_file, collect, Linkage, Status};

const USAGE: &str = "usage: tptc-verify [--root DIR] [--fail-on-unverified] PATH...
  PATH            a .tptc file, or a directory of them
  --root DIR      repo root that `source` paths are relative to (default: .)
  --fail-on-unverified
                  treat nonlinear (unverified) clauses as failures";

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut root = PathBuf::from(".");
    let mut fail_on_unverified = false;
    let mut paths = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--root" => match args.next() {
                Some(dir) => root = PathBuf::from(dir),
                None => return usage_error("--root needs a directory"),
            },
            "--fail-on-unverified" => fail_on_unverified = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            flag if flag.starts_with('-') => return usage_error(&format!("unknown flag {flag}")),
            _ => paths.push(PathBuf::from(arg)),
        }
    }
    if paths.is_empty() {
        return usage_error("no PATH given");
    }

    let mut files = Vec::new();
    for p in &paths {
        match collect(p) {
            Ok(mut f) => files.append(&mut f),
            Err(e) => {
                eprintln!("error: cannot read {}: {e}", p.display());
                return ExitCode::from(2);
            }
        }
    }

    let mut failed = 0usize;
    let (mut proven, mut unverified) = (0usize, 0usize);
    for file in &files {
        let report = match check_file(file, &root) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("error: cannot read {}: {e}", file.display());
                return ExitCode::from(2);
            }
        };
        println!("{}", file.display());
        for e in &report.parse_errors {
            println!("  PARSE ERROR  {e}");
        }
        if let Some(e) = &report.verify_error {
            println!("  VERIFY ERROR {e}");
        }
        for c in &report.clauses {
            let kind = match c.kind {
                Kind::Requires => "requires",
                Kind::Ensures => "ensures ",
            };
            let label = format!("  {kind}[{}]", c.id);
            match &c.status {
                Status::Assumed => println!("{label:<40} assumed"),
                Status::Proven => {
                    proven += 1;
                    println!("{label:<40} PROVEN");
                }
                Status::Unverified(reason) => {
                    unverified += 1;
                    println!("{label:<40} UNVERIFIED (not QF_LRA: {reason})");
                }
                Status::Disproved(model) => {
                    let cex: Vec<String> = model.iter().map(|(k, v)| format!("{k}={v}")).collect();
                    println!("{label:<40} DISPROVED  counterexample: {}", cex.join(", "));
                }
                Status::Inconclusive(why) => println!("{label:<40} INCONCLUSIVE ({why})"),
            }
        }
        match &report.linkage {
            Some(Linkage::Ok) => println!("  linkage: every clause has a runtime check"),
            Some(Linkage::NotDeclared) => {
                println!("  linkage: no `source` declared, skipped (not implemented yet)")
            }
            Some(Linkage::Missing(ids)) => {
                println!("  LINKAGE FAILED: no runtime check for {}", ids.join(", "))
            }
            Some(Linkage::SourceMissing(s)) => println!("  LINKAGE FAILED: source `{s}` not found"),
            None => {}
        }
        if report.failed(fail_on_unverified) {
            failed += 1;
        }
    }

    println!(
        "\n{} file(s) checked: {proven} clause(s) proven, {unverified} unverified, {failed} failing",
        files.len()
    );
    if failed > 0 {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

fn usage_error(msg: &str) -> ExitCode {
    eprintln!("error: {msg}\n{USAGE}");
    ExitCode::from(2)
}
