// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Seeded pseudo-random numbers for KMeans initialisation.
//!
//! SplitMix64 (Vigna, public domain). It is small, has no dependencies, and gives
//! the same sequence on every platform, so a fixed seed gives a reproducible fit.
//! It is not cryptographic, and nothing here needs it to be.

/// Deterministic 64-bit generator.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Creates a generator from a seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `0..n`. Returns 0 when `n == 0`.
    pub fn next_below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        // Multiply-high maps the 64-bit draw onto 0..n with negligible bias for
        // the sizes used here.
        ((u128::from(self.next_u64()) * n as u128) >> 64) as usize
    }

    /// Uniform `f64` in `[0, 1)` with 53 random bits.
    pub fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_differ() {
        assert_ne!(SplitMix64::new(1).next_u64(), SplitMix64::new(2).next_u64());
    }

    #[test]
    fn next_below_stays_in_range() {
        let mut r = SplitMix64::new(7);
        for n in [1_usize, 2, 3, 10, 1000] {
            for _ in 0..200 {
                assert!(r.next_below(n) < n);
            }
        }
        assert_eq!(r.next_below(0), 0);
    }

    #[test]
    fn next_unit_is_in_half_open_unit_interval() {
        let mut r = SplitMix64::new(9);
        for _ in 0..1000 {
            let u = r.next_unit();
            assert!((0.0..1.0).contains(&u));
        }
    }
}
