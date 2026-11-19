// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Reading and writing numeric CSV: comma-separated, one sample per line, optional header.

/// A numeric matrix read from CSV, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Table {
    /// Number of samples.
    pub rows: usize,
    /// Number of features.
    pub cols: usize,
    /// Row-major values, `rows × cols` long.
    pub data: Vec<f64>,
}

/// Parses CSV text. Blank lines are skipped. If the first non-blank line holds a field that is
/// not a number, it is taken as a header and skipped. Every later field must be a number, and
/// every row must have the same number of fields.
pub fn parse(text: &str) -> Result<Table, String> {
    let mut data = Vec::new();
    let mut cols: Option<usize> = None;
    let mut rows = 0;
    let mut first = true;
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        let parsed: Result<Vec<f64>, _> = fields.iter().map(|f| f.parse::<f64>()).collect();
        let values = match parsed {
            Ok(v) => v,
            Err(_) if first => {
                first = false;
                continue; // a header row
            }
            Err(_) => {
                return Err(format!("line {}: a field is not a number", line_no + 1));
            }
        };
        first = false;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(format!("line {}: a value is NaN or infinite", line_no + 1));
        }
        match cols {
            None => cols = Some(values.len()),
            Some(c) if c != values.len() => {
                return Err(format!(
                    "line {}: {} fields, expected {}",
                    line_no + 1,
                    values.len(),
                    c
                ));
            }
            Some(_) => {}
        }
        data.extend(values);
        rows += 1;
    }
    match cols {
        Some(cols) if rows > 0 => Ok(Table { rows, cols, data }),
        _ => Err("the CSV has no data rows".to_string()),
    }
}

/// Parses a CSV with one value per line, such as a target column.
pub fn parse_column(text: &str) -> Result<Vec<f64>, String> {
    let table = parse(text)?;
    if table.cols != 1 {
        return Err(format!("expected one column, found {}", table.cols));
    }
    Ok(table.data)
}

/// Writes rows of values as CSV, one row per line, with the shortest exact decimal form.
pub fn format_rows(rows: usize, cols: usize, data: &[f64]) -> String {
    let mut out = String::new();
    for r in 0..rows {
        let line: Vec<String> = data[r * cols..(r + 1) * cols]
            .iter()
            .map(|v| v.to_string())
            .collect();
        out.push_str(&line.join(","));
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_numbers_and_skips_a_header() {
        let t = parse("a,b\n1,2\n\n3.5,-4\n").unwrap();
        assert_eq!((t.rows, t.cols), (2, 2));
        assert_eq!(t.data, vec![1.0, 2.0, 3.5, -4.0]);
    }

    #[test]
    fn a_bad_field_after_the_header_is_an_error_with_its_line() {
        let err = parse("1,2\n3,x\n").unwrap_err();
        assert!(err.contains("line 2"), "{err}");
    }

    #[test]
    fn ragged_rows_and_empty_files_are_errors() {
        assert!(parse("1,2\n3\n").is_err());
        assert!(parse("").is_err());
        assert!(parse("a,b\n").is_err());
    }

    #[test]
    fn non_finite_values_are_refused() {
        assert!(parse("1,nan\n").is_err());
        assert!(parse("1,inf\n").is_err());
    }

    #[test]
    fn output_round_trips_exactly() {
        let data = [0.1_f64, 1.0 / 3.0, -2.5, 1e-300];
        let text = format_rows(2, 2, &data);
        let back = parse(&text).unwrap();
        assert_eq!(back.data, data.to_vec());
    }

    #[test]
    fn a_target_must_be_one_column() {
        assert_eq!(parse_column("1\n2\n").unwrap(), vec![1.0, 2.0]);
        assert!(parse_column("1,2\n").is_err());
    }
}
