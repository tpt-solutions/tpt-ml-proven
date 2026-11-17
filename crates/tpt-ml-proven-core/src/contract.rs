// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Contract clauses: `requires!`, `ensures!` and `static_require!`.
//!
//! * [`requires!`](crate::requires) and [`ensures!`](crate::ensures) are
//!   **runtime** checks for data-dependent clauses. They return
//!   `Err(Error::Contract(..))` from the enclosing function when the condition
//!   fails. They are active when the `runtime-contracts` feature is enabled
//!   (on by default) and compile to nothing otherwise, while the condition is
//!   still type-checked.
//! * [`static_require!`](crate::static_require) is a **compile-time** check for
//!   conditions over constants and const generics. It fails the build when the
//!   condition is false for the instantiation being compiled.
//!
//! The clause text is passed as a string literal so that failures name the
//! clause that broke, which is also what the `.tptc` contracts in
//! `docs/contracts/` refer to.

/// Whether runtime contract checks are compiled in. Controlled by the
/// `runtime-contracts` feature of this crate.
///
/// Exposed so that the macros expand to a constant that is resolved in *this*
/// crate, not in the calling crate.
pub const RUNTIME_CONTRACTS: bool = cfg!(feature = "runtime-contracts");

/// Checks a precondition. Returns `Err` from the enclosing function if `cond`
/// is false. The enclosing function must return a `Result` whose error type
/// implements `From<ContractViolation>` (for example [`crate::Error`]).
///
/// ```
/// use tpt_ml_proven_core::{requires, Error};
///
/// fn take_positive(k: usize) -> Result<usize, Error> {
///     requires!(k >= 1, "k >= 1");
///     Ok(k)
/// }
///
/// assert!(take_positive(3).is_ok());
/// # #[cfg(feature = "runtime-contracts")]
/// assert!(matches!(take_positive(0), Err(Error::Contract(_))));
/// ```
#[macro_export]
macro_rules! requires {
    ($cond:expr, $name:literal $(,)?) => {
        $crate::__contract_check!($crate::error::ContractKind::Requires, $cond, $name)
    };
}

/// Checks a postcondition. Same mechanics as [`requires!`](crate::requires), but
/// reports the clause as [`ContractKind::Ensures`](crate::error::ContractKind::Ensures),
/// meaning the implementation (not the caller) is at fault.
#[macro_export]
macro_rules! ensures {
    ($cond:expr, $name:literal $(,)?) => {
        $crate::__contract_check!($crate::error::ContractKind::Ensures, $cond, $name)
    };
}

/// Compile-time check. Use inside a function body; works with const generics.
///
/// A false condition for a given instantiation is a compile error:
///
/// ```compile_fail
/// use tpt_ml_proven_core::static_require;
///
/// fn needs_k_at_least_one<const K: usize>() {
///     static_require!(K >= 1, "K >= 1");
/// }
///
/// needs_k_at_least_one::<0>();
/// ```
///
/// ```
/// use tpt_ml_proven_core::static_require;
///
/// fn needs_k_at_least_one<const K: usize>() -> usize {
///     static_require!(K >= 1, "K >= 1");
///     K
/// }
///
/// assert_eq!(needs_k_at_least_one::<3>(), 3);
/// ```
#[macro_export]
macro_rules! static_require {
    ($cond:expr, $msg:literal $(,)?) => {
        const { ::core::assert!($cond, $msg) };
    };
}

/// Implementation detail of [`requires!`](crate::requires) and
/// [`ensures!`](crate::ensures). Not part of the public API.
#[doc(hidden)]
#[macro_export]
macro_rules! __contract_check {
    ($kind:expr, $cond:expr, $name:literal) => {
        if $crate::contract::RUNTIME_CONTRACTS && !($cond) {
            return ::core::result::Result::Err(
                $crate::error::ContractViolation {
                    kind: $kind,
                    name: $name,
                }
                .into(),
            );
        }
    };
}
