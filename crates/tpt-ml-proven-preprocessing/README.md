# tpt-ml-proven-preprocessing

Preprocessing transformers for [tpt-ml-proven](https://github.com/tpt-solutions/tpt-ml-proven).

> Status: scalers, encoders and imputers are implemented. The scalers are checked against
> scikit-learn; the encoders and imputers are checked by unit tests.

| Transformer | What it does | Checked against |
| --- | --- | --- |
| `StandardScaler` | `(x − mean) / std` per column, population std | scikit-learn, 1e-9 |
| `MinMaxScaler` | scales each column to `[0, 1]` | scikit-learn, 1e-9 |
| `RobustScaler` | `(x − median) / IQR` per column | scikit-learn, 1e-9 |
| `OneHotEncoder` | one indicator column per category | unit tests |
| `OrdinalEncoder` | category to an integer index | unit tests |
| `SimpleImputer` | fills NaN with mean, median, most frequent or a constant | unit tests |

Every transformer implements the core `UnsupervisedEstimator` and `Transformer` traits, so
`fit`, `transform` and `fit_transform` work the same way for all of them.

## Example

```rust
use tpt_ml_proven_core::{DenseView, Transformer, UnsupervisedEstimator};
use tpt_ml_proven_preprocessing::StandardScaler;

let x = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]; // 3 rows x 2 columns

let mut scaler = StandardScaler::<f64>::new();
scaler.fit(DenseView::new(&x, 3, 2).unwrap()).unwrap();
let z = scaler.transform(DenseView::new(&x, 3, 2).unwrap()).unwrap();
let back = scaler.inverse_transform(&z).unwrap();
```

## Conventions

These follow scikit-learn:

- Statistics are computed in `f64`, whatever the element type. Results are rounded to the
  element type.
- The standard deviation is the population one (`ddof = 0`).
- A zero scale or range is replaced by 1, so the transform stays finite.

## Encoders and imputers

- `OneHotEncoder` and `OrdinalEncoder` take categories given as numbers. An unknown category
  at transform time is an error, not a silent zero row.
- `SimpleImputer` treats NaN as missing. The fill value is learned from the non-missing values
  of each column during `fit`.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Licence

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache-2.0](../../LICENSE-APACHE), at
your option.
