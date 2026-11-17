// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Parser for `.tptc` contract files.
//!
//! File format (one clause per line; `#` starts a comment):
//!
//! ```text
//! contract kmeans                  # required, once, first
//! source crates/…/src/lib.rs       # optional: file that must carry the runtime checks
//! requires[rows_positive]: rows >= 1
//! ensures[k_le_rows]: k <= rows
//! ```
//!
//! Expressions: numbers, identifiers (may contain `.`, e.g. `X.rows`), `+ - * /`,
//! parentheses, relations `<= < >= > == !=`, and booleans `! && || =>`.
//! Function application `f(x)` parses but is treated as uninterpreted.

use std::fmt;

/// A parsed contract file.
#[derive(Debug, Clone, PartialEq)]
pub struct Contract {
    /// Contract name from the `contract` header. Used as the prefix of runtime
    /// check names (`"<name>.<id>"`).
    pub name: String,
    /// Repo-relative path of the Rust file that implements the runtime checks.
    pub source: Option<String>,
    /// Clauses in file order.
    pub clauses: Vec<Clause>,
}

/// Whether a clause is a precondition or a postcondition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// `requires`: assumed when checking postconditions.
    Requires,
    /// `ensures`: must follow from the requires clauses.
    Ensures,
}

/// One `requires[id]:` or `ensures[id]:` line.
#[derive(Debug, Clone, PartialEq)]
pub struct Clause {
    /// Kind of clause.
    pub kind: Kind,
    /// Stable identifier, unique within the contract.
    pub id: String,
    /// The clause text as written (after the colon, trimmed).
    pub text: String,
    /// Parsed expression.
    pub expr: Expr,
    /// 1-based line number in the source file.
    pub line: usize,
}

/// Arithmetic operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
}

/// Relation operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelOp {
    /// `<=`
    Le,
    /// `<`
    Lt,
    /// `>=`
    Ge,
    /// `>`
    Gt,
    /// `==`
    Eq,
    /// `!=`
    Ne,
}

/// A clause expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Numeric literal, kept as written.
    Num(String),
    /// Variable.
    Var(String),
    /// Unary minus.
    Neg(Box<Expr>),
    /// Arithmetic.
    Arith(ArithOp, Box<Expr>, Box<Expr>),
    /// Uninterpreted function application.
    Call(String, Vec<Expr>),
    /// Relation.
    Rel(RelOp, Box<Expr>, Box<Expr>),
    /// Boolean negation.
    Not(Box<Expr>),
    /// Conjunction.
    And(Box<Expr>, Box<Expr>),
    /// Disjunction.
    Or(Box<Expr>, Box<Expr>),
    /// Implication.
    Implies(Box<Expr>, Box<Expr>),
}

/// A syntax error with its 1-based line number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// 1-based line number.
    pub line: usize,
    /// What went wrong.
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ParseError {}

/// Parses the text of a `.tptc` file.
pub fn parse_contract(text: &str) -> Result<Contract, Vec<ParseError>> {
    let mut name: Option<String> = None;
    let mut source: Option<String> = None;
    let mut clauses: Vec<Clause> = Vec::new();
    let mut errors = Vec::new();

    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = match raw.find('#') {
            Some(p) => &raw[..p],
            None => raw,
        }
        .trim();
        if line.is_empty() {
            continue;
        }

        if let Some(rest) = line.strip_prefix("contract ") {
            let n = rest.trim();
            if !is_ident(n) {
                errors.push(err(line_no, format!("invalid contract name `{n}`")));
            } else if name.is_some() {
                errors.push(err(line_no, "duplicate `contract` header".into()));
            } else {
                name = Some(n.to_string());
            }
        } else if let Some(rest) = line.strip_prefix("source ") {
            let path = rest.trim();
            if path.is_empty() {
                errors.push(err(line_no, "empty `source` path".into()));
            } else if source.is_some() {
                errors.push(err(line_no, "duplicate `source` line".into()));
            } else {
                source = Some(path.to_string());
            }
        } else {
            let (kind, after) = if let Some(r) = line.strip_prefix("requires[") {
                (Kind::Requires, r)
            } else if let Some(r) = line.strip_prefix("ensures[") {
                (Kind::Ensures, r)
            } else {
                errors.push(err(
                    line_no,
                    "expected `contract`, `source`, `requires[id]:` or `ensures[id]:`".into(),
                ));
                continue;
            };
            let Some(close) = after.find(']') else {
                errors.push(err(line_no, "missing `]` after clause id".into()));
                continue;
            };
            let id = after[..close].trim();
            if !is_ident(id) {
                errors.push(err(line_no, format!("invalid clause id `{id}`")));
                continue;
            }
            let Some(body) = after[close + 1..].trim_start().strip_prefix(':') else {
                errors.push(err(line_no, "expected `:` after clause id".into()));
                continue;
            };
            let text = body.trim();
            if clauses.iter().any(|c| c.id == id) {
                errors.push(err(line_no, format!("duplicate clause id `{id}`")));
                continue;
            }
            match parse_expr(text) {
                Ok(expr) if !crate::smt::is_proposition(&expr) => errors.push(err(
                    line_no,
                    format!("clause `{id}` must be a comparison or a boolean combination of comparisons"),
                )),
                Ok(expr) => clauses.push(Clause {
                    kind,
                    id: id.to_string(),
                    text: text.to_string(),
                    expr,
                    line: line_no,
                }),
                Err(msg) => errors.push(err(line_no, format!("in clause `{id}`: {msg}"))),
            }
        }
    }

    if name.is_none() {
        errors.push(err(1, "missing `contract <name>` header".into()));
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(Contract {
        name: name.unwrap_or_default(),
        source,
        clauses,
    })
}

fn err(line: usize, message: String) -> ParseError {
    ParseError { line, message }
}

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

// ---------------------------------------------------------------- lexer

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(String),
    Ident(String),
    Sym(&'static str),
}

const SYMS: [&str; 17] = [
    "<=", ">=", "==", "!=", "&&", "||", "=>", "<", ">", "=", "+", "-", "*", "/", "(", ")", ",",
];

fn lex(s: &str) -> Result<Vec<Tok>, String> {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() {
            let start = i;
            while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                i += 1;
            }
            if i < bytes.len() && bytes[i] == b'.' {
                i += 1;
                let frac = i;
                while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                    i += 1;
                }
                if i == frac {
                    return Err("number ends with `.`".into());
                }
            }
            out.push(Tok::Num(s[start..i].to_string()));
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < bytes.len() {
                let d = bytes[i] as char;
                if d.is_ascii_alphanumeric() || d == '_' || d == '.' {
                    i += 1;
                } else {
                    break;
                }
            }
            out.push(Tok::Ident(s[start..i].to_string()));
        } else if c == '!' && bytes.get(i + 1) != Some(&b'=') {
            out.push(Tok::Sym("!"));
            i += 1;
        } else {
            match SYMS.iter().find(|sym| s[i..].starts_with(*sym)) {
                Some(sym) => {
                    out.push(Tok::Sym(sym));
                    i += sym.len();
                }
                None => return Err(format!("unexpected character `{c}`")),
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------- parser

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

/// Parses a single clause expression.
pub fn parse_expr(text: &str) -> Result<Expr, String> {
    let toks = lex(text)?;
    if toks.is_empty() {
        return Err("empty expression".into());
    }
    let mut p = Parser { toks, pos: 0 };
    let e = p.implies()?;
    if p.pos != p.toks.len() {
        return Err(format!("unexpected token {:?}", p.toks[p.pos]));
    }
    Ok(e)
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn eat(&mut self, sym: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Sym(s)) if *s == sym) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn implies(&mut self) -> Result<Expr, String> {
        let lhs = self.or()?;
        if self.eat("=>") {
            let rhs = self.implies()?;
            return Ok(Expr::Implies(Box::new(lhs), Box::new(rhs)));
        }
        Ok(lhs)
    }

    fn or(&mut self) -> Result<Expr, String> {
        let mut lhs = self.and()?;
        while self.eat("||") {
            let rhs = self.and()?;
            lhs = Expr::Or(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn and(&mut self) -> Result<Expr, String> {
        let mut lhs = self.unary_bool()?;
        while self.eat("&&") {
            let rhs = self.unary_bool()?;
            lhs = Expr::And(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn unary_bool(&mut self) -> Result<Expr, String> {
        if self.eat("!") {
            return Ok(Expr::Not(Box::new(self.unary_bool()?)));
        }
        self.relation()
    }

    fn relation(&mut self) -> Result<Expr, String> {
        let lhs = self.arith()?;
        let op = match self.peek() {
            Some(Tok::Sym("<=")) => RelOp::Le,
            Some(Tok::Sym("<")) => RelOp::Lt,
            Some(Tok::Sym(">=")) => RelOp::Ge,
            Some(Tok::Sym(">")) => RelOp::Gt,
            Some(Tok::Sym("==")) | Some(Tok::Sym("=")) => RelOp::Eq,
            Some(Tok::Sym("!=")) => RelOp::Ne,
            _ => return Ok(lhs),
        };
        self.pos += 1;
        let rhs = self.arith()?;
        Ok(Expr::Rel(op, Box::new(lhs), Box::new(rhs)))
    }

    fn arith(&mut self) -> Result<Expr, String> {
        let mut lhs = self.term()?;
        loop {
            let op = if self.eat("+") {
                ArithOp::Add
            } else if self.eat("-") {
                ArithOp::Sub
            } else {
                break;
            };
            let rhs = self.term()?;
            lhs = Expr::Arith(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn term(&mut self) -> Result<Expr, String> {
        let mut lhs = self.unary()?;
        loop {
            let op = if self.eat("*") {
                ArithOp::Mul
            } else if self.eat("/") {
                ArithOp::Div
            } else {
                break;
            };
            let rhs = self.unary()?;
            lhs = Expr::Arith(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<Expr, String> {
        if self.eat("-") {
            return Ok(Expr::Neg(Box::new(self.unary()?)));
        }
        self.primary()
    }

    fn primary(&mut self) -> Result<Expr, String> {
        match self.peek().cloned() {
            Some(Tok::Num(n)) => {
                self.pos += 1;
                Ok(Expr::Num(n))
            }
            Some(Tok::Ident(name)) => {
                self.pos += 1;
                if self.eat("(") {
                    let mut args = Vec::new();
                    if !self.eat(")") {
                        loop {
                            args.push(self.implies()?);
                            if self.eat(")") {
                                break;
                            }
                            if !self.eat(",") {
                                return Err("expected `,` or `)` in call".into());
                            }
                        }
                    }
                    Ok(Expr::Call(name, args))
                } else {
                    Ok(Expr::Var(name))
                }
            }
            Some(Tok::Sym("(")) => {
                self.pos += 1;
                let e = self.implies()?;
                if !self.eat(")") {
                    return Err("missing `)`".into());
                }
                Ok(e)
            }
            Some(t) => Err(format!("unexpected token {t:?}")),
            None => Err("unexpected end of expression".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_header_source_and_clauses() {
        let c = parse_contract(
            "# comment\ncontract kmeans\nsource crates/x/src/lib.rs\n\
             requires[rows_positive]: rows >= 1\nensures[k_le]: k <= rows  # trailing\n",
        )
        .unwrap();
        assert_eq!(c.name, "kmeans");
        assert_eq!(c.source.as_deref(), Some("crates/x/src/lib.rs"));
        assert_eq!(c.clauses.len(), 2);
        assert_eq!(c.clauses[0].kind, Kind::Requires);
        assert_eq!(c.clauses[1].id, "k_le");
        assert_eq!(c.clauses[1].line, 5);
    }

    #[test]
    fn precedence_and_grouping() {
        // a + b * 2 <= c  parses as  (a + (b * 2)) <= c
        let e = parse_expr("a + b * 2 <= c").unwrap();
        match e {
            Expr::Rel(RelOp::Le, lhs, _) => match *lhs {
                Expr::Arith(ArithOp::Add, _, rhs) => {
                    assert!(matches!(*rhs, Expr::Arith(ArithOp::Mul, _, _)))
                }
                other => panic!("unexpected lhs {other:?}"),
            },
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn implication_is_right_associative() {
        let e = parse_expr("a >= 0 => b >= 0 => c >= 0").unwrap();
        assert!(matches!(e, Expr::Implies(_, rhs) if matches!(*rhs, Expr::Implies(_, _))));
    }

    #[test]
    fn bang_and_not_equal_are_distinguished() {
        assert!(matches!(parse_expr("!(a >= 0)").unwrap(), Expr::Not(_)));
        assert!(matches!(
            parse_expr("a != b").unwrap(),
            Expr::Rel(RelOp::Ne, _, _)
        ));
    }

    #[test]
    fn reports_every_line_error() {
        let errs = parse_contract(
            "contract c\nbogus\nrequires[x]: a >=\nensures[x]: a >= 0\nrequires[x]: b >= 0\n",
        )
        .unwrap_err();
        let lines: Vec<usize> = errs.iter().map(|e| e.line).collect();
        assert_eq!(lines, vec![2, 3, 5]);
        assert!(errs[2].message.contains("duplicate"));
    }

    #[test]
    fn missing_header_is_an_error() {
        let errs = parse_contract("requires[a]: x >= 0\n").unwrap_err();
        assert!(errs.iter().any(|e| e.message.contains("missing `contract")));
    }

    #[test]
    fn calls_and_decimals_parse() {
        assert!(matches!(
            parse_expr("inertia(s1) <= inertia(s0) + 0.5").unwrap(),
            Expr::Rel(RelOp::Le, _, _)
        ));
        assert!(parse_expr("x >= 1.").is_err());
    }
}
