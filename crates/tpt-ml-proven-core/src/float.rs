// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Generic floating-point abstraction over `f32` and `f64`.
//!
//! Only the operations the estimators need are included. Anything needing
//! `libm` (for example `sqrt`) is deliberately absent, so the core crate has no
//! runtime dependencies and stays `no_std` without extra code.

use core::fmt::Debug;
use core::ops::{Add, Div, Mul, Neg, Sub};

/// A floating-point scalar usable by the estimators (`f32` or `f64`).
pub trait Float:
    Copy
    + Debug
    + PartialOrd
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;
    /// Machine epsilon for this type.
    const EPSILON: Self;

    /// Converts a count to this type. Values above 2^24 (`f32`) or 2^53 (`f64`)
    /// lose precision.
    fn from_usize(n: usize) -> Self;

    /// `true` if the value is neither infinite nor NaN.
    fn is_finite(self) -> bool;
}

macro_rules! impl_float {
    ($t:ty) => {
        impl Float for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const EPSILON: Self = <$t>::EPSILON;

            #[inline]
            fn from_usize(n: usize) -> Self {
                n as $t
            }

            #[inline]
            fn is_finite(self) -> bool {
                <$t>::is_finite(self)
            }
        }
    };
}

impl_float!(f32);
impl_float!(f64);

#[cfg(test)]
mod tests {
    use super::Float;

    fn sum_all<F: Float>(xs: &[F]) -> F {
        xs.iter().fold(F::ZERO, |acc, &x| acc + x)
    }

    #[test]
    fn generic_arithmetic_matches_for_both_widths() {
        assert_eq!(sum_all::<f32>(&[1.0, 2.0, 3.0]), 6.0);
        assert_eq!(sum_all::<f64>(&[1.0, 2.0, 3.0]), 6.0);
    }

    #[test]
    fn from_usize_and_finiteness() {
        assert_eq!(<f64 as Float>::from_usize(7), 7.0);
        assert!(Float::is_finite(1.5_f32));
        assert!(!Float::is_finite(f64::NAN));
        assert!(!Float::is_finite(f32::INFINITY));
    }
}
