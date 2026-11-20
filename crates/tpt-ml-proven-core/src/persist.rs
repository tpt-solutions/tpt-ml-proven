// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! The on-disk model format, version 2. Version 1 files still load.
//!
//! A model file is one byte string. Version 2 has a dtype tag after the kind:
//!
//! ```text
//! offset  size  field
//! 0       8     magic            b"TPTMLMDL"
//! 8       4     format version   u32, little-endian, currently 2
//! 12      4     model kind       u32, little-endian (see [`kind`])
//! 16      4     dtype            u32, little-endian: 1 = f32, 2 = f64 (see [`dtype`])
//! 20      …     payload          fields, in the order the model writes them
//! end-8   8     checksum         u64, little-endian, FNV-1a (64-bit) of every byte before it
//! ```
//!
//! Version 1 has no dtype field: its payload starts at offset 16. Its models are reported as
//! [`dtype::F64`], the representation they were widened to.
//!
//! Payload fields are little-endian: `u64` for counts and indices, `f64` for values (`f32`
//! models are widened on write and rounded back on read), one byte for `bool`, and
//! `u64` length followed by the values for slices. A stored `f64` must be finite: a NaN or
//! infinity in the file is an error.
//!
//! # Compatibility rules
//!
//! * A reader accepts any version from 1 up to [`FORMAT_VERSION`]. A newer file is rejected
//!   with an error, never guessed at.
//! * The kind must match the type being loaded.
//! * Within a version, fields are never reordered or removed. A change to a model's fields
//!   needs a new version and a reader for the old one.
//! * The checksum catches truncation and corruption. It is not a signature: a file can be
//!   altered and re-checksummed, so treat model files as trusted input.

use alloc::vec::Vec;

use crate::error::Error;

/// Dtype tags stored in the version 2 header.
pub mod dtype {
    /// Single precision: the model was fitted over `f32`.
    pub const F32: u32 = 1;
    /// Double precision: the model was fitted over `f64`. Version 1 files report this too.
    pub const F64: u32 = 2;
}

/// The eight magic bytes at the start of every model file.
pub const MAGIC: [u8; 8] = *b"TPTMLMDL";

/// The format version this crate writes, and the newest it reads.
pub const FORMAT_VERSION: u32 = 2;

/// Model kind tags, stored in the header so a file cannot be loaded as the wrong type.
pub mod kind {
    /// `KMeans`.
    pub const KMEANS: u32 = 1;
    /// `Pca`.
    pub const PCA: u32 = 2;
    /// `Ridge`.
    pub const RIDGE: u32 = 3;
    /// `Lasso`.
    pub const LASSO: u32 = 4;
    /// `ElasticNet`.
    pub const ELASTIC_NET: u32 = 5;
    /// `LogisticRegression` (binary).
    pub const LOGISTIC: u32 = 6;
    /// `MulticlassLogistic`.
    pub const MULTICLASS_LOGISTIC: u32 = 7;
    /// `DecisionTreeRegressor`.
    pub const DECISION_TREE: u32 = 8;
    /// `GradientBoostingRegressor`.
    pub const GRADIENT_BOOSTING: u32 = 9;
    /// `TruncatedSvd`.
    pub const TRUNCATED_SVD: u32 = 10;
    /// `Nmf`.
    pub const NMF: u32 = 11;
    /// `StumpForest`.
    pub const STUMP_FOREST: u32 = 12;
    /// `StandardScaler`.
    pub const STANDARD_SCALER: u32 = 13;
    /// `MinMaxScaler`.
    pub const MINMAX_SCALER: u32 = 14;
    /// `RobustScaler`.
    pub const ROBUST_SCALER: u32 = 15;
    /// `OneHotEncoder`.
    pub const ONE_HOT_ENCODER: u32 = 16;
    /// `OrdinalEncoder`.
    pub const ORDINAL_ENCODER: u32 = 17;
    /// `SimpleImputer`.
    pub const SIMPLE_IMPUTER: u32 = 18;
    /// `DecisionTreeClassifier`.
    pub const DECISION_TREE_CLASSIFIER: u32 = 19;
    /// `RandomForestRegressor`.
    pub const RANDOM_FOREST_REGRESSOR: u32 = 20;
    /// `RandomForestClassifier`.
    pub const RANDOM_FOREST_CLASSIFIER: u32 = 21;
    /// `GradientBoostingRegressor` with subsample, max_features and seed (kind 9 has none).
    pub const GRADIENT_BOOSTING_SAMPLED: u32 = 22;
    /// `Dbscan`.
    pub const DBSCAN: u32 = 23;
    /// `Hdbscan`.
    pub const HDBSCAN: u32 = 24;
    /// A binary `LogisticRegression` with the two labels it was trained on.
    pub const LOGISTIC_LABELLED: u32 = 25;
    /// A `DecisionTreeRegressor` that stores its leaf limit, split size and feature importances.
    /// Files of kind `DECISION_TREE` (8) remain readable.
    pub const DECISION_TREE_V2: u32 = 26;
    /// A `DecisionTreeClassifier` that stores its leaf limit, split size and feature importances.
    /// Files of kind `DECISION_TREE_CLASSIFIER` (19) remain readable.
    pub const DECISION_TREE_CLASSIFIER_V2: u32 = 27;
    /// A `RandomForestRegressor` that stores its split size, leaf limit and feature importances.
    /// Files of kind `RANDOM_FOREST_REGRESSOR` (20) remain readable.
    pub const RANDOM_FOREST_REGRESSOR_V2: u32 = 28;
    /// A `RandomForestClassifier` that stores its split size, leaf limit and feature importances.
    /// Files of kind `RANDOM_FOREST_CLASSIFIER` (21) remain readable.
    pub const RANDOM_FOREST_CLASSIFIER_V2: u32 = 29;
    /// A `GradientBoostingRegressor` that stores its split size, leaf limit and feature
    /// importances. Files of kind `GRADIENT_BOOSTING_SAMPLED` (22) remain readable.
    pub const GRADIENT_BOOSTING_V2: u32 = 30;
}

/// Header length of a version 1 file: magic, version, kind.
const HEADER_LEN_V1: usize = 16;
/// Header length of a version 2 file: the version 1 header plus the dtype.
const HEADER_LEN_V2: usize = 20;
const TRAILER_LEN: usize = 8;

/// FNV-1a, 64-bit. Used for the checksum only.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn format_error(reason: &'static str) -> Error {
    Error::Format { reason }
}

/// Builds a model file, field by field.
#[derive(Debug)]
pub struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    /// Starts a file for a model of the given `kind` and dtype, with the header written.
    pub fn with_dtype(model_kind: u32, model_dtype: u32) -> Self {
        let mut buf = Vec::with_capacity(64);
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&model_kind.to_le_bytes());
        buf.extend_from_slice(&model_dtype.to_le_bytes());
        Self { buf }
    }

    /// Starts a file for a model over float type `F`, tagged with the matching dtype.
    pub fn for_float<F: crate::Float>(model_kind: u32) -> Self {
        let tag = if core::mem::size_of::<F>() == 4 {
            dtype::F32
        } else {
            dtype::F64
        };
        Self::with_dtype(model_kind, tag)
    }

    /// Writes a count or index.
    pub fn usize(&mut self, v: usize) {
        self.u64(v as u64);
    }

    /// Writes a 64-bit unsigned integer, such as a seed.
    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Writes a value.
    pub fn f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Writes a flag as one byte.
    pub fn bool(&mut self, v: bool) {
        self.buf.push(u8::from(v));
    }

    /// Writes a length, then each value.
    pub fn f64s(&mut self, values: &[f64]) {
        self.usize(values.len());
        for &v in values {
            self.f64(v);
        }
    }

    /// Writes a length, then each index.
    pub fn usizes(&mut self, values: &[usize]) {
        self.usize(values.len());
        for &v in values {
            self.usize(v);
        }
    }

    /// Writes a length, then the bytes themselves.
    pub fn bytes(&mut self, values: &[u8]) {
        self.usize(values.len());
        self.buf.extend_from_slice(values);
    }

    /// Writes a length, then each row as a length-prefixed slice of values.
    pub fn f64_rows(&mut self, rows: &[Vec<f64>]) {
        self.usize(rows.len());
        for row in rows {
            self.f64s(row);
        }
    }

    /// Appends the checksum and returns the file.
    pub fn finish(mut self) -> Vec<u8> {
        let sum = fnv1a(&self.buf);
        self.buf.extend_from_slice(&sum.to_le_bytes());
        self.buf
    }
}

/// Reads a model file, checking the header and checksum first.
#[derive(Debug)]
pub struct Reader<'a> {
    payload: &'a [u8],
    pos: usize,
    dtype: u32,
}

impl<'a> Reader<'a> {
    /// Checks the header, the version, the kind and the checksum, and returns a reader over
    /// the payload. Version 1 and version 2 files are both accepted.
    pub fn new(bytes: &'a [u8], expected_kind: u32) -> Result<Self, Error> {
        if bytes.len() < HEADER_LEN_V1 + TRAILER_LEN {
            return Err(format_error("file is shorter than the header and checksum"));
        }
        if bytes[..8] != MAGIC {
            return Err(format_error("not a tpt-ml-proven model file"));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes"));
        let (header_len, file_dtype) = match version {
            1 => (HEADER_LEN_V1, dtype::F64),
            2 => {
                if bytes.len() < HEADER_LEN_V2 + TRAILER_LEN {
                    return Err(format_error("file is shorter than the header and checksum"));
                }
                let tag = u32::from_le_bytes(bytes[16..20].try_into().expect("4 bytes"));
                if tag != dtype::F32 && tag != dtype::F64 {
                    return Err(format_error("model file has an unknown dtype tag"));
                }
                (HEADER_LEN_V2, tag)
            }
            _ => {
                return Err(format_error(
                    "model file was written by a newer or unknown format version",
                ))
            }
        };
        let kind = u32::from_le_bytes(bytes[12..16].try_into().expect("4 bytes"));
        if kind != expected_kind {
            return Err(format_error("model file holds a different kind of model"));
        }
        let body_end = bytes.len() - TRAILER_LEN;
        let stored = u64::from_le_bytes(bytes[body_end..].try_into().expect("8 bytes"));
        if fnv1a(&bytes[..body_end]) != stored {
            return Err(format_error(
                "checksum mismatch: the model file is corrupt or truncated",
            ));
        }
        Ok(Self {
            payload: &bytes[header_len..body_end],
            pos: 0,
            dtype: file_dtype,
        })
    }

    /// The dtype tag, [`dtype::F32`] or [`dtype::F64`]. Version 1 files report `F64`.
    pub fn dtype(&self) -> u32 {
        self.dtype
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.payload.len())
            .ok_or_else(|| format_error("model file ends before all fields were read"))?;
        let out = &self.payload[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    /// Reads a count or index.
    pub fn usize(&mut self) -> Result<usize, Error> {
        let raw = u64::from_le_bytes(self.take(8)?.try_into().expect("8 bytes"));
        usize::try_from(raw).map_err(|_| format_error("count does not fit in usize"))
    }

    /// Reads a 64-bit unsigned integer.
    pub fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }

    /// Reads a value. A stored NaN or infinity is an error.
    pub fn f64(&mut self) -> Result<f64, Error> {
        let v = f64::from_le_bytes(self.take(8)?.try_into().expect("8 bytes"));
        if v.is_finite() {
            Ok(v)
        } else {
            Err(format_error("model file holds a NaN or infinite value"))
        }
    }

    /// Reads a flag.
    pub fn bool(&mut self) -> Result<bool, Error> {
        match self.take(1)?[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(format_error("flag is neither 0 nor 1")),
        }
    }

    /// Reads a length, then that many values. The length is checked against the bytes left,
    /// so a corrupt length cannot cause a huge allocation.
    pub fn f64s(&mut self) -> Result<Vec<f64>, Error> {
        let n = self.usize()?;
        if n > (self.payload.len() - self.pos) / 8 {
            return Err(format_error("slice length exceeds the data in the file"));
        }
        (0..n).map(|_| self.f64()).collect()
    }

    /// Reads a length, then that many indices.
    pub fn usizes(&mut self) -> Result<Vec<usize>, Error> {
        let n = self.usize()?;
        if n > (self.payload.len() - self.pos) / 8 {
            return Err(format_error("slice length exceeds the data in the file"));
        }
        (0..n).map(|_| self.usize()).collect()
    }

    /// Reads a length, then that many bytes.
    pub fn bytes(&mut self) -> Result<Vec<u8>, Error> {
        let n = self.usize()?;
        if n > self.payload.len() - self.pos {
            return Err(format_error("byte string exceeds the data in the file"));
        }
        Ok(self.take(n)?.to_vec())
    }

    /// Reads a length, then that many length-prefixed rows of values.
    pub fn f64_rows(&mut self) -> Result<Vec<Vec<f64>>, Error> {
        let n = self.usize()?;
        // Every row costs at least its 8-byte length, so this bounds the allocation.
        if n > (self.payload.len() - self.pos) / 8 {
            return Err(format_error("row count exceeds the data in the file"));
        }
        (0..n).map(|_| self.f64s()).collect()
    }

    /// Checks that every payload byte was read. Call last.
    pub fn finish(self) -> Result<(), Error> {
        if self.pos == self.payload.len() {
            Ok(())
        } else {
            Err(format_error(
                "model file has bytes left after the last field",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn sample() -> Vec<u8> {
        let mut w = Writer::with_dtype(kind::RIDGE, dtype::F64);
        w.usize(3);
        w.f64(-1.5);
        w.bool(true);
        w.f64s(&[1.0, 2.0]);
        w.usizes(&[4, 5, 6]);
        w.finish()
    }

    #[test]
    fn round_trip_recovers_every_field() {
        let bytes = sample();
        let mut r = Reader::new(&bytes, kind::RIDGE).unwrap();
        assert_eq!(r.usize().unwrap(), 3);
        assert_eq!(r.f64().unwrap(), -1.5);
        assert!(r.bool().unwrap());
        assert_eq!(r.f64s().unwrap(), vec![1.0, 2.0]);
        assert_eq!(r.usizes().unwrap(), vec![4, 5, 6]);
        r.finish().unwrap();
    }

    #[test]
    fn version_two_header_is_pinned() {
        let bytes = sample();
        assert_eq!(&bytes[..8], b"TPTMLMDL");
        assert_eq!(&bytes[8..12], &2_u32.to_le_bytes());
        assert_eq!(&bytes[12..16], &kind::RIDGE.to_le_bytes());
        assert_eq!(&bytes[16..20], &dtype::F64.to_le_bytes());
    }

    #[test]
    fn wrong_kind_and_corruption_are_rejected() {
        let bytes = sample();
        assert!(Reader::new(&bytes, kind::PCA).is_err());
        let mut flipped = bytes.clone();
        flipped[20] ^= 0x01;
        assert!(Reader::new(&flipped, kind::RIDGE).is_err());
        assert!(Reader::new(&bytes[..bytes.len() - 1], kind::RIDGE).is_err());
        assert!(Reader::new(b"not a model at all, really", kind::RIDGE).is_err());
    }

    #[test]
    fn newer_format_version_is_refused() {
        let mut w = Writer::with_dtype(kind::RIDGE, dtype::F64);
        w.usize(0);
        let mut bytes = w.finish();
        bytes[8..12].copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
        // Re-checksum so only the version is wrong.
        let body_end = bytes.len() - 8;
        let sum = fnv1a(&bytes[..body_end]);
        bytes[body_end..].copy_from_slice(&sum.to_le_bytes());
        assert!(matches!(
            Reader::new(&bytes, kind::RIDGE),
            Err(Error::Format { .. })
        ));
    }

    #[test]
    fn huge_declared_length_does_not_allocate() {
        let mut w = Writer::with_dtype(kind::RIDGE, dtype::F64);
        w.usize(usize::MAX / 16);
        let bytes = w.finish();
        let mut r = Reader::new(&bytes, kind::RIDGE).unwrap();
        assert!(r.f64s().is_err());
    }

    /// A version 1 file as written before the dtype tag existed: no dtype field.
    fn version_one_file(payload: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&kind::RIDGE.to_le_bytes());
        bytes.extend_from_slice(payload);
        let sum = fnv1a(&bytes);
        bytes.extend_from_slice(&sum.to_le_bytes());
        bytes
    }

    #[test]
    fn version_one_files_still_load_and_report_f64() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&2.5_f64.to_le_bytes());
        let bytes = version_one_file(&payload);
        let mut r = Reader::new(&bytes, kind::RIDGE).unwrap();
        assert_eq!(r.dtype(), dtype::F64);
        assert_eq!(r.f64().unwrap(), 2.5);
        r.finish().unwrap();
    }

    #[test]
    fn dtype_tag_follows_the_float_type() {
        let w = Writer::for_float::<f32>(kind::RIDGE);
        let bytes = w.finish();
        assert_eq!(
            Reader::new(&bytes, kind::RIDGE).unwrap().dtype(),
            dtype::F32
        );
        let w = Writer::for_float::<f64>(kind::RIDGE);
        let bytes = w.finish();
        assert_eq!(
            Reader::new(&bytes, kind::RIDGE).unwrap().dtype(),
            dtype::F64
        );
    }

    #[test]
    fn unknown_dtype_tag_is_refused() {
        let mut bytes = Writer::with_dtype(kind::RIDGE, dtype::F64).finish();
        bytes[16..20].copy_from_slice(&7_u32.to_le_bytes());
        let body_end = bytes.len() - 8;
        let sum = fnv1a(&bytes[..body_end]);
        bytes[body_end..].copy_from_slice(&sum.to_le_bytes());
        assert!(matches!(
            Reader::new(&bytes, kind::RIDGE),
            Err(Error::Format { .. })
        ));
    }

    #[test]
    fn nan_and_infinity_in_the_file_are_refused() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut w = Writer::with_dtype(kind::RIDGE, dtype::F64);
            w.f64(bad);
            let bytes = w.finish();
            let mut r = Reader::new(&bytes, kind::RIDGE).unwrap();
            assert!(r.f64().is_err(), "{bad} should be refused");
        }
    }

    #[test]
    fn rows_round_trip() {
        let rows = vec![vec![1.0, 2.0], vec![], vec![3.5]];
        let mut w = Writer::with_dtype(kind::RIDGE, dtype::F64);
        w.f64_rows(&rows);
        let bytes = w.finish();
        let mut r = Reader::new(&bytes, kind::RIDGE).unwrap();
        assert_eq!(r.f64_rows().unwrap(), rows);
        r.finish().unwrap();
    }
}
