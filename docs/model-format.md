# Model file format, version 2

This is the on-disk format for fitted models. It is implemented in
`crates/tpt-ml-proven-core/src/persist.rs`, and this document is the reference for it.
Other implementations, in other languages, should follow this page.

## Why our own format

The roadmap first planned serde with bincode behind a feature flag. We use our own format
instead, for three reasons:

* **Stability.** Field order and encoding are fixed here, in writing, and checked by golden
  files. A serde derive follows the struct layout, so an ordinary refactor could change the
  bytes without anyone noticing.
* **No dependencies.** The core crate is `no_std` + `alloc` with no dependencies, and this
  format keeps it that way. Bincode would have added one.
* **Other languages.** The format is simple enough to read from Go, Python or JavaScript
  without a library.

## Layout

All integers and floats are little-endian.

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 8 | Magic, the ASCII bytes `TPTMLMDL` |
| 8 | 4 | Format version, `u32`. Version 2 is `2` |
| 12 | 4 | Model kind, `u32` (table below) |
| 16 | 4 | Dtype, `u32`: `1` = f32, `2` = f64 (version 2 only) |
| 20 | … | Payload: the model's fields, in the order the model writes them |
| end − 8 | 8 | Checksum, `u64`: FNV-1a (64-bit) over every byte before it |

Version 1 files have no dtype field, so their payload starts at offset 16. A version 1 file
reads as f64, the representation its values were widened to.

### Field encodings

| Type | Encoding |
| --- | --- |
| count, index (`usize`) | `u64` |
| 64-bit integer (`u64`, e.g. a seed) | `u64` |
| value (`f64`) | IEEE 754 binary64. Must be finite: a NaN or infinity is an error on load |
| flag (`bool`) | one byte: `0` or `1`. Any other value is an error |
| slice of values | `u64` length, then that many values |
| slice of indices | `u64` length, then that many `u64` |
| matrix of values | `u64` row count, then each row as a slice of values |

`f32` models are widened to `f64` on write and rounded back on read. The dtype field records
which float type the model was fitted over, so a reader knows the precision the values came
from.

### Model kinds

| Kind | Value | Model |
| --- | --- | --- |
| `KMEANS` | 1 | `KMeans` |
| `PCA` | 2 | `Pca` |
| `RIDGE` | 3 | `Ridge` |
| `LASSO` | 4 | `Lasso` |
| `ELASTIC_NET` | 5 | `ElasticNet` |
| `LOGISTIC` | 6 | `LogisticRegression` (binary) |
| `MULTICLASS_LOGISTIC` | 7 | `MulticlassLogistic` |
| `DECISION_TREE` | 8 | `DecisionTreeRegressor` |
| `GRADIENT_BOOSTING` | 9 | `GradientBoostingRegressor` |
| `TRUNCATED_SVD` | 10 | `TruncatedSvd` |
| `NMF` | 11 | `Nmf` |
| `STUMP_FOREST` | 12 | `StumpForest` |
| `STANDARD_SCALER` | 13 | `StandardScaler` |
| `MINMAX_SCALER` | 14 | `MinMaxScaler` |
| `ROBUST_SCALER` | 15 | `RobustScaler` |
| `ONE_HOT_ENCODER` | 16 | `OneHotEncoder` |
| `ORDINAL_ENCODER` | 17 | `OrdinalEncoder` |
| `SIMPLE_IMPUTER` | 18 | `SimpleImputer` |
| `DECISION_TREE_CLASSIFIER` | 19 | `DecisionTreeClassifier` (without the newer settings) |
| `RANDOM_FOREST_REGRESSOR` | 20 | `RandomForestRegressor` |
| `RANDOM_FOREST_CLASSIFIER` | 21 | `RandomForestClassifier` |
| `GRADIENT_BOOSTING_SAMPLED` | 22 | `GradientBoostingRegressor` with subsampling or feature sampling |
| `DBSCAN` | 23 | `Dbscan` |
| `HDBSCAN` | 24 | `Hdbscan` |
| `LOGISTIC_LABELLED` | 25 | `LogisticRegression` (binary), with the two labels it was trained on |
| `DECISION_TREE_V2` | 26 | `DecisionTreeRegressor` with a split size, a leaf limit and importances |
| `DECISION_TREE_CLASSIFIER_V2` | 27 | `DecisionTreeClassifier` with a split size, a leaf limit and importances |

A model's payload starts with its parameters, then a flag for whether it is fitted, then the
fitted state. A loaded model can be refitted with the same settings.

A model that gains fields gets a new kind rather than a new version, so the files already written
keep loading. Kinds 8 and 19 are the single trees without the newer settings, and kinds 26 and 27
add the split size, the leaf limit, and the per-feature importances. A model that uses none of the
newer settings is still written in the older kind, so loading and saving an older file leaves its
bytes unchanged. Kinds 8 and 19 load with the defaults and without importances.

## Compatibility rules

1. A reader accepts any version from 1 up to the newest it knows. A file with a newer version
   is refused with an error. It is never read on a guess.
2. The kind must match the model being loaded. A ridge file cannot load as a lasso.
3. Within one version, fields are never reordered, removed or re-typed.
4. A change to a model's fields needs a new format version. The reader for the old version
   stays, so old files keep loading. Version 2 added the dtype field and the finite-value rule;
   version 1 files still load.
5. Every declared length is checked against the bytes that remain, so a corrupt length cannot
   cause a huge allocation.
6. Loading also checks each model's invariants (shapes, index ranges, parameter ranges). A file
   that passes the checksum but is inconsistent is refused.

The checksum detects truncation and corruption. It is not a signature. Anyone can change a
file and recompute the checksum, so only load model files from sources you trust.

## Golden files

Each committed golden file is a version 1 model that a test loads on every run:

* `crates/tpt-ml-proven-linear/tests/fixtures/ridge_v1.tpml`
* `crates/tpt-ml-proven-clustering/tests/fixtures/kmeans_v1.tpml`
* `crates/tpt-ml-proven-decomposition/tests/fixtures/pca_v1.tpml`

These are version 1 files, kept to prove that version 1 still loads. Version 2 is pinned by a
header test in `persist.rs`.

Version 2 golden files, one per model kind added since version 1. A test loads each file and
checks that saving it back gives the same bytes, which pins the layout. Each is written by an
ignored `regenerate_golden` test (run with `--ignored`), never by hand:

* `crates/tpt-ml-proven-decomposition/tests/fixtures/svd_v2.tpml`, `nmf_v2.tpml`
* `crates/tpt-ml-proven-ensemble/tests/fixtures/stump_forest_v2.tpml`
* `crates/tpt-ml-proven-preprocessing/tests/fixtures/` (`standard_scaler_v2`, `minmax_scaler_v2`,
  `robust_scaler_v2`, `one_hot_v2`, `ordinal_v2`, `simple_imputer_v2`, each `.tpml`) If one of these tests fails after a change, the change has broken
existing files. Bump the format version instead of editing the fixture.

## Worked example: the header

The header of a version 2 file for an f64 ridge model starts with:

```text
54 50 54 4d 4c 4d 44 4c   "TPTMLMDL"
02 00 00 00               format version 2
03 00 00 00               kind 3 (ridge)
02 00 00 00               dtype 2 (f64)
```

## Scope

Rust has the reader and the writer. The Python, WebAssembly, C and Go bindings do not yet
expose save and load. The format is ready for them, and they will use these same bytes.
