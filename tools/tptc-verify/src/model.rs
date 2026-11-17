// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Exact evaluation of clauses on a concrete assignment.
//!
//! The solver's SAT answers are never trusted on their own. A counterexample is
//! reported only if this module confirms it against the original clauses. The
//! solver has been seen to return a model that violates a strict inequality, so
//! this check is not optional.

use std::collections::HashMap;

use crate::parse::{ArithOp, Expr, RelOp};

/// Tolerance for float comparison. Model values are exact rationals converted to
/// `f64`, so the error is tiny; this margin absorbs it.
pub const TOL: f64 = 1e-9;

/// Parses a solver value: a decimal (`2.5`), an integer (`-3`) or a fraction
/// (`1/3`).
pub fn parse_value(s: &str) -> Option<f64> {
    if let Some((n, d)) = s.split_once('/') {
        let n: f64 = n.trim().parse().ok()?;
        let d: f64 = d.trim().parse().ok()?;
        if d == 0.0 {
            return None;
        }
        return Some(n / d);
    }
    s.trim().parse().ok()
}

/// Evaluates an arithmetic term. Variables missing from `env` read as zero, which
/// is safe: any assignment that falsifies a clause is a valid counterexample.
pub fn eval_num(e: &Expr, env: &HashMap<String, f64>) -> Option<f64> {
    match e {
        Expr::Num(n) => n.parse().ok(),
        Expr::Var(v) => Some(*env.get(v).unwrap_or(&0.0)),
        Expr::Neg(a) => Some(-eval_num(a, env)?),
        Expr::Arith(op, a, b) => {
            let (x, y) = (eval_num(a, env)?, eval_num(b, env)?);
            match op {
                ArithOp::Add => Some(x + y),
                ArithOp::Sub => Some(x - y),
                ArithOp::Mul => Some(x * y),
                ArithOp::Div if y != 0.0 => Some(x / y),
                ArithOp::Div => None,
            }
        }
        // Uninterpreted functions have no meaning here, so the model cannot be
        // confirmed against them.
        Expr::Call(_, _) => None,
        _ => None,
    }
}

/// Evaluates a clause leniently: each relation may be off by [`TOL`] in the
/// direction that helps it hold. Used for `requires`, so a counterexample is
/// never rejected only because it sits on a boundary.
pub fn holds_lenient(e: &Expr, env: &HashMap<String, f64>) -> Option<bool> {
    holds(e, env, Mode::Lenient)
}

/// Evaluates a clause robustly: each relation must hold with a margin of [`TOL`].
/// Used on the negated `ensures`, so a disproof means the postcondition is
/// violated, not merely touched.
pub fn holds_robust(e: &Expr, env: &HashMap<String, f64>) -> Option<bool> {
    holds(e, env, Mode::Robust)
}

#[derive(Clone, Copy)]
enum Mode {
    Lenient,
    Robust,
}

fn holds(e: &Expr, env: &HashMap<String, f64>, mode: Mode) -> Option<bool> {
    match e {
        Expr::Rel(op, a, b) => {
            let (l, r) = (eval_num(a, env)?, eval_num(b, env)?);
            let d = l - r;
            Some(match (mode, op) {
                (Mode::Lenient, RelOp::Le) => d <= TOL,
                (Mode::Lenient, RelOp::Lt) => d < TOL,
                (Mode::Lenient, RelOp::Ge) => d >= -TOL,
                (Mode::Lenient, RelOp::Gt) => d > -TOL,
                (Mode::Lenient, RelOp::Eq) => d.abs() <= TOL,
                (Mode::Lenient, RelOp::Ne) => true,
                (Mode::Robust, RelOp::Le) => d <= -TOL,
                (Mode::Robust, RelOp::Lt) => d < -TOL,
                (Mode::Robust, RelOp::Ge) => d >= TOL,
                (Mode::Robust, RelOp::Gt) => d > TOL,
                (Mode::Robust, RelOp::Eq) => d.abs() <= TOL,
                (Mode::Robust, RelOp::Ne) => d.abs() > TOL,
            })
        }
        Expr::Not(a) => Some(!holds(a, env, mode)?),
        Expr::And(a, b) => Some(holds(a, env, mode)? && holds(b, env, mode)?),
        Expr::Or(a, b) => Some(holds(a, env, mode)? || holds(b, env, mode)?),
        Expr::Implies(a, b) => Some(!holds(a, env, mode)? || holds(b, env, mode)?),
        _ => None,
    }
}

/// Logical negation, pushed to the atoms. Used to build the robust check of a
/// postcondition's violation.
pub fn negate(e: &Expr) -> Expr {
    match e {
        Expr::Rel(op, a, b) => {
            let op = match op {
                RelOp::Le => RelOp::Gt,
                RelOp::Lt => RelOp::Ge,
                RelOp::Ge => RelOp::Lt,
                RelOp::Gt => RelOp::Le,
                RelOp::Eq => RelOp::Ne,
                RelOp::Ne => RelOp::Eq,
            };
            Expr::Rel(op, a.clone(), b.clone())
        }
        Expr::Not(a) => (**a).clone(),
        Expr::And(a, b) => Expr::Or(Box::new(negate(a)), Box::new(negate(b))),
        Expr::Or(a, b) => Expr::And(Box::new(negate(a)), Box::new(negate(b))),
        Expr::Implies(a, b) => Expr::And(a.clone(), Box::new(negate(b))),
        other => Expr::Not(Box::new(other.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_expr;

    fn env(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn parses_decimals_integers_and_fractions() {
        assert_eq!(parse_value("2.5"), Some(2.5));
        assert_eq!(parse_value("-3"), Some(-3.0));
        assert!((parse_value("1/4").unwrap() - 0.25).abs() < 1e-15);
        assert_eq!(parse_value("1/0"), None);
        assert_eq!(parse_value("abc"), None);
    }

    #[test]
    fn robust_violation_needs_a_margin() {
        let neg = negate(&parse_expr("x >= 1").unwrap());
        // x = 1 touches the boundary: not a robust violation of x >= 1.
        assert_eq!(holds_robust(&neg, &env(&[("x", 1.0)])), Some(false));
        assert_eq!(holds_robust(&neg, &env(&[("x", 0.5)])), Some(true));
    }

    #[test]
    fn lenient_hypothesis_accepts_boundary_models() {
        let h = parse_expr("x >= 1").unwrap();
        assert_eq!(holds_lenient(&h, &env(&[("x", 1.0)])), Some(true));
        assert_eq!(holds_lenient(&h, &env(&[("x", 0.0)])), Some(false));
    }

    #[test]
    fn missing_variables_read_as_zero_and_calls_are_unknown() {
        let e = parse_expr("y >= 0").unwrap();
        assert_eq!(holds_lenient(&e, &env(&[])), Some(true));
        assert_eq!(
            holds_lenient(&parse_expr("f(y) >= 0").unwrap(), &env(&[])),
            None
        );
    }

    #[test]
    fn negation_is_exact_on_points() {
        let g = parse_expr("a < b || a == 2").unwrap();
        let ng = negate(&g);
        // Concrete: a=3,b=1 falsifies `a < b || a == 2`, and its negation holds.
        let en = env(&[("a", 3.0), ("b", 1.0)]);
        assert_eq!(holds_lenient(&g, &en), Some(false));
        assert_eq!(holds_lenient(&ng, &en), Some(true));
    }
}
