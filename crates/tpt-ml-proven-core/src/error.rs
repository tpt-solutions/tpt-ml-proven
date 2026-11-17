// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Shared error types: shape errors, fit errors and contract violations.

use core::fmt;

/// A matrix was built or used with the wrong shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeError {
    /// Row count that was supplied.
    pub rows: usize,
    /// Column count that was supplied.
    pub cols: usize,
    /// What was wrong, as a fixed description.
    pub reason: &'static str,
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "shape error ({} x {}): {}",
            self.rows, self.cols, self.reason
        )
    }
}

/// Whether a violated contract was a precondition or a postcondition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractKind {
    /// A `requires` clause: the caller broke it.
    Requires,
    /// An `ensures` clause: the implementation broke it (a bug).
    Ensures,
}

/// A `requires!` or `ensures!` clause did not hold at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContractViolation {
    /// Which kind of clause failed.
    pub kind: ContractKind,
    /// The clause's name, as written in the source.
    pub name: &'static str,
}

impl fmt::Display for ContractViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            ContractKind::Requires => "precondition",
            ContractKind::Ensures => "postcondition",
        };
        write!(f, "contract violation ({kind}): {}", self.name)
    }
}

/// Errors returned by estimator `fit` and by `predict` / `transform`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// Input had the wrong shape.
    Shape(ShapeError),
    /// Input had no rows or no columns where at least one of each is needed.
    EmptyInput,
    /// Input contained NaN or infinity.
    NonFinite {
        /// Which input was non-finite.
        what: &'static str,
    },
    /// A hyper-parameter was out of its valid range.
    InvalidParameter {
        /// Parameter name.
        name: &'static str,
        /// Why the value was rejected.
        reason: &'static str,
    },
    /// `predict` or `transform` was called before `fit`.
    NotFitted,
    /// A contract clause failed at runtime.
    Contract(ContractViolation),
}

/// Error type returned by `fit`. Same as [`Error`].
pub type FitError = Error;

impl From<ShapeError> for Error {
    fn from(e: ShapeError) -> Self {
        Error::Shape(e)
    }
}

impl From<ContractViolation> for Error {
    fn from(e: ContractViolation) -> Self {
        Error::Contract(e)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Shape(e) => fmt::Display::fmt(e, f),
            Error::EmptyInput => write!(f, "input is empty"),
            Error::NonFinite { what } => write!(f, "{what} contains NaN or infinity"),
            Error::InvalidParameter { name, reason } => {
                write!(f, "invalid parameter `{name}`: {reason}")
            }
            Error::NotFitted => write!(f, "estimator has not been fitted"),
            Error::Contract(e) => fmt::Display::fmt(e, f),
        }
    }
}

impl core::error::Error for Error {}
impl core::error::Error for ShapeError {}
impl core::error::Error for ContractViolation {}
