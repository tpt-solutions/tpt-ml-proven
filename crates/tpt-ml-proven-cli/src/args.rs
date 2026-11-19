// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Hand-rolled argument parsing: positional words and `--name value` flags.

use std::collections::BTreeMap;
use std::str::FromStr;

/// Parsed command line: positional words in order, and named flags.
#[derive(Debug, Default, PartialEq)]
pub struct Args {
    positional: Vec<String>,
    flags: BTreeMap<String, String>,
}

impl Args {
    /// Splits `items` into positional words and `--name value` pairs.
    pub fn parse(items: &[String]) -> Result<Self, String> {
        let mut out = Self::default();
        let mut iter = items.iter();
        while let Some(item) = iter.next() {
            match item.strip_prefix("--") {
                Some("") => return Err("a bare -- is not a flag".to_string()),
                Some(name) => {
                    let value = iter
                        .next()
                        .ok_or_else(|| format!("--{name} needs a value"))?;
                    out.flags.insert(name.to_string(), value.clone());
                }
                None => out.positional.push(item.clone()),
            }
        }
        Ok(out)
    }

    /// The positional word at `index`, or an error naming what it was for.
    pub fn positional(&self, index: usize, what: &str) -> Result<&str, String> {
        self.positional
            .get(index)
            .map(String::as_str)
            .ok_or_else(|| format!("missing {what}"))
    }

    /// Number of positional words.
    pub fn positional_len(&self) -> usize {
        self.positional.len()
    }

    /// The value of a flag, if given.
    pub fn flag(&self, name: &str) -> Option<&str> {
        self.flags.get(name).map(String::as_str)
    }

    /// The value of a required flag.
    pub fn required(&self, name: &str) -> Result<&str, String> {
        self.flag(name)
            .ok_or_else(|| format!("--{name} is required"))
    }

    /// A flag parsed as `T`, or `default` when absent.
    pub fn parsed_or<T: FromStr>(&self, name: &str, default: T) -> Result<T, String> {
        match self.flag(name) {
            None => Ok(default),
            Some(text) => text
                .parse::<T>()
                .map_err(|_| format!("--{name} has an invalid value: {text}")),
        }
    }

    /// Fails if a flag was given that the command does not use.
    pub fn reject_unknown(&self, known: &[&str]) -> Result<(), String> {
        for name in self.flags.keys() {
            if !known.contains(&name.as_str()) {
                return Err(format!("unknown option --{name}"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn positionals_and_flags_are_separated() {
        let a = Args::parse(&words(&["fit", "kmeans", "--k", "3", "--seed", "7"])).unwrap();
        assert_eq!(a.positional(0, "command").unwrap(), "fit");
        assert_eq!(a.positional(1, "estimator").unwrap(), "kmeans");
        assert_eq!(a.parsed_or::<usize>("k", 8).unwrap(), 3);
        assert_eq!(a.parsed_or::<u64>("seed", 0).unwrap(), 7);
        assert_eq!(a.parsed_or::<usize>("missing", 5).unwrap(), 5);
    }

    #[test]
    fn a_flag_without_a_value_is_an_error() {
        assert!(Args::parse(&words(&["--k"])).is_err());
    }

    #[test]
    fn unknown_flags_are_rejected() {
        let a = Args::parse(&words(&["--colour", "red"])).unwrap();
        assert!(a.reject_unknown(&["k"]).is_err());
    }

    #[test]
    fn bad_numbers_name_the_flag() {
        let a = Args::parse(&words(&["--k", "three"])).unwrap();
        let err = a.parsed_or::<usize>("k", 1).unwrap_err();
        assert!(err.contains("--k"));
    }
}
