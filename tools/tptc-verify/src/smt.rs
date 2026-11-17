// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Lowering of clause expressions to conjunctive QF_LRA queries.
//!
//! The solver rejects disjunctions over arithmetic, so formulas are split into
//! disjunctive normal form and each conjunction (a *cube*) is one query. Strict
//! inequalities are sent exactly; the solver's strict handling was fixed in
//! tpt-solver (`fix/strict-lra`).

use std::collections::BTreeSet;

use crate::parse::{ArithOp, Expr, RelOp};

/// Upper bound on the number of cubes. Above it, the case split is abandoned and
/// the clause is reported inconclusive rather than run for ages.
pub const MAX_CUBES: usize = 256;

/// Polarity of a subformula during negation normal form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pol {
    /// The subformula is asserted as written.
    Pos,
    /// The subformula is negated.
    Neg,
}

/// One exact arithmetic atom in SMT-LIB2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Atom {
    /// SMT-LIB2 text of the atom.
    pub smt: String,
    /// Variables the atom mentions.
    pub vars: BTreeSet<String>,
}

/// A conjunction of atoms. An empty cube means `true`.
pub type Cube = Vec<Atom>;

/// Returns why `e` is outside QF_LRA, or `None` if it is linear.
pub fn nonlinear_reason(e: &Expr) -> Option<String> {
    match e {
        Expr::Num(_) | Expr::Var(_) => None,
        Expr::Neg(a) | Expr::Not(a) => nonlinear_reason(a),
        Expr::Arith(op, a, b) => match op {
            ArithOp::Add | ArithOp::Sub => nonlinear_reason(a).or_else(|| nonlinear_reason(b)),
            ArithOp::Mul => {
                if is_const(a) || is_const(b) {
                    nonlinear_reason(a).or_else(|| nonlinear_reason(b))
                } else {
                    Some("product of two non-constant terms".into())
                }
            }
            ArithOp::Div => {
                if is_const(b) {
                    nonlinear_reason(a)
                } else {
                    Some("division by a non-constant term".into())
                }
            }
        },
        Expr::Call(name, _) => Some(format!("uninterpreted function `{name}`")),
        Expr::Rel(_, a, b) | Expr::And(a, b) | Expr::Or(a, b) | Expr::Implies(a, b) => {
            nonlinear_reason(a).or_else(|| nonlinear_reason(b))
        }
    }
}

/// `true` if the arithmetic term contains no variables.
fn is_const(e: &Expr) -> bool {
    match e {
        Expr::Num(_) => true,
        Expr::Neg(a) => is_const(a),
        Expr::Arith(_, a, b) => is_const(a) && is_const(b),
        _ => false,
    }
}

/// `true` if the root of a clause is a proposition (a relation or a boolean
/// combination of relations), not a bare arithmetic term.
pub fn is_proposition(e: &Expr) -> bool {
    matches!(
        e,
        Expr::Rel(..) | Expr::Not(_) | Expr::And(..) | Expr::Or(..) | Expr::Implies(..)
    )
}

fn vars_of(e: &Expr, out: &mut BTreeSet<String>) {
    match e {
        Expr::Num(_) => {}
        Expr::Var(v) => {
            out.insert(v.clone());
        }
        Expr::Neg(a) | Expr::Not(a) => vars_of(a, out),
        Expr::Arith(_, a, b)
        | Expr::Rel(_, a, b)
        | Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Implies(a, b) => {
            vars_of(a, out);
            vars_of(b, out);
        }
        Expr::Call(_, args) => args.iter().for_each(|a| vars_of(a, out)),
    }
}

/// Renders a linear arithmetic term as SMT-LIB2. Integer literals become reals
/// (`3` becomes `3.0`) so they sort against `Real` variables.
///
/// Callers must check [`nonlinear_reason`] first.
pub fn term_to_smt(e: &Expr) -> String {
    match e {
        Expr::Num(n) if n.contains('.') => n.clone(),
        Expr::Num(n) => format!("{n}.0"),
        Expr::Var(v) => v.clone(),
        Expr::Neg(a) => format!("(- {})", term_to_smt(a)),
        Expr::Arith(op, a, b) => {
            let sym = match op {
                ArithOp::Add => "+",
                ArithOp::Sub => "-",
                ArithOp::Mul => "*",
                ArithOp::Div => "/",
            };
            format!("({sym} {} {})", term_to_smt(a), term_to_smt(b))
        }
        Expr::Call(_, _) => unreachable!("nonlinear_reason rejects calls"),
        _ => unreachable!("term_to_smt is only called on arithmetic terms"),
    }
}

/// The atom for `a op b` under polarity `pol`, or `None` for `true`. Disequalities
/// are handled by [`cubes`] as a disjunction, so they never come through here.
fn atom(op: RelOp, pol: Pol, a: &Expr, b: &Expr) -> Option<Atom> {
    let (l, r) = (term_to_smt(a), term_to_smt(b));
    let smt = match (op, pol) {
        (RelOp::Le, Pol::Pos) => format!("(<= {l} {r})"),
        (RelOp::Lt, Pol::Pos) => format!("(< {l} {r})"),
        (RelOp::Ge, Pol::Pos) => format!("(>= {l} {r})"),
        (RelOp::Gt, Pol::Pos) => format!("(> {l} {r})"),
        (RelOp::Eq, Pol::Pos) => format!("(= {l} {r})"),
        (RelOp::Le, Pol::Neg) => format!("(> {l} {r})"),
        (RelOp::Lt, Pol::Neg) => format!("(>= {l} {r})"),
        (RelOp::Ge, Pol::Neg) => format!("(< {l} {r})"),
        (RelOp::Gt, Pol::Neg) => format!("(<= {l} {r})"),
        (RelOp::Eq, Pol::Neg) | (RelOp::Ne, Pol::Pos) => return None,
        (RelOp::Ne, Pol::Neg) => format!("(= {l} {r})"),
    };
    let mut vars = BTreeSet::new();
    vars_of(a, &mut vars);
    vars_of(b, &mut vars);
    Some(Atom { smt, vars })
}

/// Negation normal form, then expanded to DNF. `None` if the result
/// would exceed [`MAX_CUBES`].
pub fn cubes(e: &Expr, pol: Pol) -> Option<Vec<Cube>> {
    match e {
        // a != b is (a < b) or (a > b).
        Expr::Rel(RelOp::Ne, a, b) if pol == Pol::Pos => union(
            vec![vec![atom(RelOp::Lt, Pol::Pos, a, b)?]],
            vec![vec![atom(RelOp::Gt, Pol::Pos, a, b)?]],
        ),
        // not (a = b) is (a < b) or (a > b).
        Expr::Rel(RelOp::Eq, a, b) if pol == Pol::Neg => union(
            vec![vec![atom(RelOp::Lt, Pol::Pos, a, b)?]],
            vec![vec![atom(RelOp::Gt, Pol::Pos, a, b)?]],
        ),
        Expr::Rel(op, a, b) => Some(vec![atom(*op, pol, a, b).into_iter().collect()]),
        Expr::Not(a) => cubes(a, flip(pol)),
        Expr::And(a, b) => match pol {
            Pol::Pos => cross(cubes(a, pol)?, cubes(b, pol)?),
            Pol::Neg => union(cubes(a, pol)?, cubes(b, pol)?),
        },
        Expr::Or(a, b) => match pol {
            Pol::Pos => union(cubes(a, pol)?, cubes(b, pol)?),
            Pol::Neg => cross(cubes(a, pol)?, cubes(b, pol)?),
        },
        Expr::Implies(a, b) => match pol {
            Pol::Pos => union(cubes(a, Pol::Neg)?, cubes(b, Pol::Pos)?),
            Pol::Neg => cross(cubes(a, Pol::Pos)?, cubes(b, Pol::Neg)?),
        },
        _ => None,
    }
}

fn flip(p: Pol) -> Pol {
    match p {
        Pol::Pos => Pol::Neg,
        Pol::Neg => Pol::Pos,
    }
}

fn union(a: Vec<Cube>, b: Vec<Cube>) -> Option<Vec<Cube>> {
    if a.len() + b.len() > MAX_CUBES {
        return None;
    }
    Some(a.into_iter().chain(b).collect())
}

fn cross(a: Vec<Cube>, b: Vec<Cube>) -> Option<Vec<Cube>> {
    if a.len().saturating_mul(b.len()) > MAX_CUBES {
        return None;
    }
    let mut out = Vec::with_capacity(a.len() * b.len());
    for x in &a {
        for y in &b {
            let mut c = x.clone();
            c.extend(y.iter().cloned());
            out.push(c);
        }
    }
    Some(out)
}

/// The queries that together decide `hyps ⊨ goal`.
///
/// Each query is one hypothesis cube conjoined with one cube of the negated goal.
/// `hyps ⊨ goal` holds when every query is UNSAT. A SAT query is a candidate
/// counterexample, and the caller must confirm it exactly (see `model.rs`).
pub fn proof_queries(hyps: &[&Expr], goal: &Expr) -> Option<Vec<Cube>> {
    let h = hypothesis_cubes(hyps)?;
    cross(h, cubes(goal, Pol::Neg)?)
}

/// The DNF of the conjunction of `hyps`. Empty `hyps` gives one empty cube.
pub fn hypothesis_cubes(hyps: &[&Expr]) -> Option<Vec<Cube>> {
    let mut acc: Vec<Cube> = vec![Vec::new()];
    for h in hyps {
        acc = cross(acc, cubes(h, Pol::Pos)?)?;
    }
    Some(acc)
}

/// Builds the SMT-LIB2 script for one cube.
pub fn script(cube: &[Atom]) -> String {
    let mut vars = BTreeSet::new();
    for a in cube {
        vars.extend(a.vars.iter().cloned());
    }
    let mut s = String::from("(set-logic QF_LRA)\n");
    for v in &vars {
        s.push_str(&format!("(declare-const {v} Real)\n"));
    }
    for a in cube {
        s.push_str(&format!("(assert {})\n", a.smt));
    }
    s.push_str("(check-sat)\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_expr;

    #[test]
    fn linear_clauses_are_accepted() {
        for text in [
            "k <= rows",
            "2 * x + y >= 3.5",
            "x / 4 - 1 < y",
            "!(a == b) && c > 0",
        ] {
            assert_eq!(nonlinear_reason(&parse_expr(text).unwrap()), None, "{text}");
        }
    }

    #[test]
    fn nonlinear_clauses_are_rejected_with_a_reason() {
        let cases = [
            ("a * b <= 1", "product"),
            ("1 / n <= 1", "division"),
            ("inertia(s) <= 3", "uninterpreted function `inertia`"),
        ];
        for (text, needle) in cases {
            let r = nonlinear_reason(&parse_expr(text).unwrap()).expect(text);
            assert!(r.contains(needle), "{text}: {r}");
        }
    }

    #[test]
    fn strict_relations_are_emitted_exactly() {
        // not (x >= 0 && y >= 0)  ==  x < 0 || y < 0  ->  two cubes, exact.
        let g = parse_expr("x >= 0 && y >= 0").unwrap();
        let cs = cubes(&g, Pol::Neg).unwrap();
        assert_eq!(cs.len(), 2);
        assert_eq!(cs[0][0].smt, "(< x 0.0)");
        assert_eq!(cs[1][0].smt, "(< y 0.0)");
    }

    #[test]
    fn disequality_splits_into_strict_cases() {
        // a != b  ==  a < b || a > b
        let e = parse_expr("a != b").unwrap();
        let cs = cubes(&e, Pol::Pos).unwrap();
        assert_eq!(cs.len(), 2);
        assert_eq!(cs[0][0].smt, "(< a b)");
        assert_eq!(cs[1][0].smt, "(> a b)");
    }

    #[test]
    fn negated_relations_are_exact() {
        let cases = [
            ("a <= b", "(> a b)"),
            ("a < b", "(>= a b)"),
            ("a >= b", "(< a b)"),
            ("a > b", "(<= a b)"),
        ];
        for (text, want) in cases {
            let cs = cubes(&parse_expr(text).unwrap(), Pol::Neg).unwrap();
            assert_eq!(cs[0][0].smt, want, "not ({text})");
        }
    }

    #[test]
    fn disjunctive_hypothesis_multiplies_queries() {
        let h = parse_expr("x <= 0 || x >= 2").unwrap();
        let g = parse_expr("x >= 1").unwrap();
        let qs = proof_queries(&[&h], &g).unwrap();
        // 2 hypothesis cubes x 1 negated-goal cube.
        assert_eq!(qs.len(), 2);
    }

    #[test]
    fn case_split_is_capped() {
        // 9 binary disjunctions -> 512 cubes, over the cap.
        let text = (0..9)
            .map(|i| format!("(x{i} >= 0 || x{i} <= 1)"))
            .collect::<Vec<_>>()
            .join(" && ");
        let h = parse_expr(&text).unwrap();
        assert!(hypothesis_cubes(&[&h]).is_none());
    }

    #[test]
    fn script_declares_only_its_variables_and_asserts_the_cube() {
        let h = parse_expr("rows >= 1").unwrap();
        let cs = hypothesis_cubes(&[&h]).unwrap();
        let s = script(&cs[0]);
        assert!(s.starts_with("(set-logic QF_LRA)\n"));
        assert!(s.contains("(declare-const rows Real)"));
        assert!(s.contains("(assert (>= rows 1.0))"));
        assert!(s.ends_with("(check-sat)\n"));
    }
}
