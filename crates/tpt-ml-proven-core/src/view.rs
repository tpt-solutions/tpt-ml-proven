// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Input views: [`ArrayView`] for feature matrices and [`TargetView`] for labels.
//!
//! Estimators take views, so a caller's existing buffer can be passed in without
//! a copy. [`DenseView`] borrows a row-major slice. [`Dense`] owns one and is
//! the output type of transformers.

use alloc::vec::Vec;

use crate::error::ShapeError;
use crate::float::Float;

/// A read-only 2-D matrix of `Float` values, indexed by `(row, col)`.
pub trait ArrayView {
    /// Element type.
    type Elem: Float;

    /// Number of rows (samples).
    fn rows(&self) -> usize;

    /// Number of columns (features).
    fn cols(&self) -> usize;

    /// Returns the element at `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics if `row >= self.rows()` or `col >= self.cols()`.
    fn get(&self, row: usize, col: usize) -> Self::Elem;

    /// Returns row `i` as a contiguous slice when the layout allows it, so hot
    /// loops can avoid per-element indexing. Returns `None` otherwise.
    fn row_slice(&self, _i: usize) -> Option<&[Self::Elem]> {
        None
    }

    /// `true` if every element is finite.
    fn all_finite(&self) -> bool {
        for r in 0..self.rows() {
            for c in 0..self.cols() {
                if !self.get(r, c).is_finite() {
                    return false;
                }
            }
        }
        true
    }
}

/// A read-only 1-D target vector (labels or regression targets).
pub trait TargetView {
    /// Element type.
    type Elem: Copy;

    /// Number of targets.
    fn len(&self) -> usize;

    /// `true` if there are no targets.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the target at index `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.len()`.
    fn get(&self, i: usize) -> Self::Elem;
}

impl<T: Copy> TargetView for &[T] {
    type Elem = T;

    fn len(&self) -> usize {
        <[T]>::len(self)
    }

    fn get(&self, i: usize) -> T {
        self[i]
    }
}

fn checked_len(rows: usize, cols: usize, actual: usize) -> Result<(), ShapeError> {
    match rows.checked_mul(cols) {
        Some(n) if n == actual => Ok(()),
        Some(_) => Err(ShapeError {
            rows,
            cols,
            reason: "buffer length does not equal rows * cols",
        }),
        None => Err(ShapeError {
            rows,
            cols,
            reason: "rows * cols overflows usize",
        }),
    }
}

impl<T: ArrayView + ?Sized> ArrayView for &T {
    type Elem = T::Elem;

    fn rows(&self) -> usize {
        T::rows(self)
    }

    fn cols(&self) -> usize {
        T::cols(self)
    }

    fn get(&self, row: usize, col: usize) -> T::Elem {
        T::get(self, row, col)
    }

    fn row_slice(&self, i: usize) -> Option<&[T::Elem]> {
        T::row_slice(self, i)
    }
}

/// Borrowed, row-major view over a slice. No copy is made.
#[derive(Debug, Clone, Copy)]
pub struct DenseView<'a, T: Float> {
    data: &'a [T],
    rows: usize,
    cols: usize,
}

impl<'a, T: Float> DenseView<'a, T> {
    /// Wraps a row-major slice of length `rows * cols`.
    pub fn new(data: &'a [T], rows: usize, cols: usize) -> Result<Self, ShapeError> {
        checked_len(rows, cols, data.len())?;
        Ok(Self { data, rows, cols })
    }
}

impl<T: Float> ArrayView for DenseView<'_, T> {
    type Elem = T;

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn get(&self, row: usize, col: usize) -> T {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        self.data[row * self.cols + col]
    }

    fn row_slice(&self, i: usize) -> Option<&[T]> {
        if i < self.rows {
            Some(&self.data[i * self.cols..(i + 1) * self.cols])
        } else {
            None
        }
    }
}

/// Owned, row-major matrix. Used as the output of transformers.
#[derive(Debug, Clone, PartialEq)]
pub struct Dense<T: Float> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: Float> Dense<T> {
    /// Takes ownership of a row-major buffer of length `rows * cols`.
    pub fn from_vec(data: Vec<T>, rows: usize, cols: usize) -> Result<Self, ShapeError> {
        checked_len(rows, cols, data.len())?;
        Ok(Self { data, rows, cols })
    }

    /// Returns a borrowed view of this matrix.
    pub fn view(&self) -> DenseView<'_, T> {
        DenseView {
            data: &self.data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Consumes the matrix and returns the row-major buffer.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T: Float> ArrayView for Dense<T> {
    type Elem = T;

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn get(&self, row: usize, col: usize) -> T {
        self.view().get(row, col)
    }

    fn row_slice(&self, i: usize) -> Option<&[T]> {
        self.view().row_slice(i).map(|_| {
            let start = i * self.cols;
            &self.data[start..start + self.cols]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_view_indexes_row_major() {
        let buf = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = DenseView::new(&buf, 2, 3).unwrap();
        assert_eq!((v.rows(), v.cols()), (2, 3));
        assert_eq!(v.get(0, 2), 3.0);
        assert_eq!(v.get(1, 0), 4.0);
        assert_eq!(v.row_slice(1), Some(&buf[3..6]));
        assert_eq!(v.row_slice(2), None);
    }

    #[test]
    fn wrong_length_is_a_shape_error() {
        let buf = [1.0_f32; 5];
        let err = DenseView::new(&buf, 2, 3).unwrap_err();
        assert_eq!((err.rows, err.cols), (2, 3));
    }

    #[test]
    fn overflowing_shape_is_a_shape_error() {
        let buf: [f64; 0] = [];
        assert!(DenseView::new(&buf, usize::MAX, 2).is_err());
    }

    #[test]
    fn dense_round_trips_and_detects_non_finite() {
        let m = Dense::from_vec(alloc::vec![1.0_f32, f32::NAN, 3.0, 4.0], 2, 2).unwrap();
        assert!(!m.all_finite());
        let ok = Dense::from_vec(alloc::vec![1.0_f32, 2.0], 1, 2).unwrap();
        assert!(ok.all_finite());
        assert_eq!(ok.into_vec(), alloc::vec![1.0, 2.0]);
    }

    #[test]
    fn targets_index_and_len() {
        let y: &[i32] = &[7, 8, 9];
        assert_eq!(TargetView::len(&y), 3);
        assert_eq!(TargetView::get(&y, 1), 8);
        assert!(!TargetView::is_empty(&y));
    }
}
