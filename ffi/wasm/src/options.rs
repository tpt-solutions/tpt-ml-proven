// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Reading constructor options from a JavaScript object.
//!
//! Every constructor takes one optional object with camelCase keys, for example
//! `new KMeansModel({ nClusters: 3, seed: 7 })`. Missing or `undefined` keys take their
//! defaults; a key with the wrong type is an error that names the key. Counts and seeds must be
//! non-negative integers no larger than `Number.MAX_SAFE_INTEGER`, so that they are exact.

use wasm_bindgen::{JsError, JsValue};

/// The largest integer a JavaScript number holds exactly.
const MAX_SAFE: f64 = 9_007_199_254_740_991.0;

/// A constructor's options object, or nothing.
pub(crate) struct Options {
    object: Option<JsValue>,
}

impl Options {
    /// Accepts `undefined`, `null` or an object. Anything else is an error.
    pub(crate) fn new(value: Option<JsValue>) -> Result<Self, JsError> {
        match value {
            None => Ok(Self { object: None }),
            Some(v) if v.is_undefined() || v.is_null() => Ok(Self { object: None }),
            Some(v) if v.is_object() => Ok(Self { object: Some(v) }),
            Some(_) => Err(JsError::new(
                "options must be an object, for example { nClusters: 3 }",
            )),
        }
    }

    fn raw(&self, key: &str) -> Result<Option<JsValue>, JsError> {
        let Some(object) = &self.object else {
            return Ok(None);
        };
        let value = js_sys::Reflect::get(object, &JsValue::from_str(key))
            .map_err(|_| JsError::new(&format!("cannot read option {key}")))?;
        Ok(if value.is_undefined() || value.is_null() {
            None
        } else {
            Some(value)
        })
    }

    /// A number, or `None` when the key is absent.
    pub(crate) fn f64_opt(&self, key: &str) -> Result<Option<f64>, JsError> {
        match self.raw(key)? {
            None => Ok(None),
            Some(v) => v
                .as_f64()
                .map(Some)
                .ok_or_else(|| JsError::new(&format!("option {key} must be a number"))),
        }
    }

    /// A number, or `default` when the key is absent.
    pub(crate) fn f64_or(&self, key: &str, default: f64) -> Result<f64, JsError> {
        Ok(self.f64_opt(key)?.unwrap_or(default))
    }

    /// A non-negative integer, or `None` when the key is absent.
    pub(crate) fn usize_opt(&self, key: &str) -> Result<Option<usize>, JsError> {
        match self.f64_opt(key)? {
            None => Ok(None),
            Some(x) if x.fract() == 0.0 && (0.0..=MAX_SAFE).contains(&x) => Ok(Some(x as usize)),
            Some(_) => Err(JsError::new(&format!(
                "option {key} must be a non-negative integer no larger than Number.MAX_SAFE_INTEGER"
            ))),
        }
    }

    /// A non-negative integer, or `default` when the key is absent.
    pub(crate) fn usize_or(&self, key: &str, default: usize) -> Result<usize, JsError> {
        Ok(self.usize_opt(key)?.unwrap_or(default))
    }

    /// A required non-negative integer.
    pub(crate) fn required_usize(&self, key: &str) -> Result<usize, JsError> {
        self.usize_opt(key)?
            .ok_or_else(|| JsError::new(&format!("option {key} is required")))
    }

    /// A seed: a non-negative integer, given as a number (not a BigInt).
    pub(crate) fn seed_or(&self, key: &str, default: u64) -> Result<u64, JsError> {
        Ok(self.usize_opt(key)?.map_or(default, |v| v as u64))
    }

    /// A string, or `default` when the key is absent.
    pub(crate) fn str_or(&self, key: &str, default: &str) -> Result<String, JsError> {
        match self.raw(key)? {
            None => Ok(default.to_string()),
            Some(v) => v
                .as_string()
                .ok_or_else(|| JsError::new(&format!("option {key} must be a string"))),
        }
    }
}
