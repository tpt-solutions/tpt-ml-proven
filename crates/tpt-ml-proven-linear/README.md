# tpt-ml-proven-linear

Linear models for [tpt-ml-proven](https://github.com/tpt-solutions/tpt-ml-proven).

> Status: `Ridge`, `Lasso`, `ElasticNet`, `LogisticRegression` (binary) and
> `MulticlassLogistic` are implemented and tested against scikit-learn.

| Estimator | Solver | Contract | scikit-learn parity |
| --- | --- | --- | --- |
| `Ridge` | closed form (Cholesky) | [`ridge.tptc`](../../docs/contracts/ridge.tptc) | coefficients within 1e-6 |
| `Lasso` | coordinate descent | [`elasticnet.tptc`](../../docs/contracts/elasticnet.tptc) | coefficients within 1e-6 |
| `ElasticNet` | coordinate descent | [`elasticnet.tptc`](../../docs/contracts/elasticnet.tptc) | coefficients within 1e-6 |
| `LogisticRegression` | damped Newton | [`logistic.tptc`](../../docs/contracts/logistic.tptc) | coefficients and probabilities within 1e-6 |
| `MulticlassLogistic` | damped Newton (softmax) | [`multiclass-logistic.tptc`](../../docs/contracts/multiclass-logistic.tptc) | coefficients and probabilities within 1e-6 |

The objectives are scikit-learn's, so the coefficients can be compared directly:

- `Ridge`: `‖y − Xw − b‖² + α‖w‖²`.
- `Lasso` and `ElasticNet`: `(1/2n)‖y − Xw − b‖² + α·ρ‖w‖₁ + ½·α·(1 − ρ)‖w‖²`. Lasso is `ρ = 1`.
- `LogisticRegression` and `MulticlassLogistic`: `½‖W‖² + C · Σ logloss`, with `C` as in
  scikit-learn. The intercepts are not penalised.

## Example

```rust
use tpt_ml_proven_core::{DenseView, Predictor, SupervisedEstimator};
use tpt_ml_proven_linear::Ridge;

let x = vec![0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0, 2.0]; // 4 rows x 2 columns
let y = vec![1.0, 2.0, 3.0, 1.5];

let mut model = Ridge::<f64>::new(0.1);
model.fit(DenseView::new(&x, 4, 2).unwrap(), &y[..]).unwrap();
let predictions = model.predict(DenseView::new(&x, 4, 2).unwrap()).unwrap();
```

## Logistic regression

`LogisticRegression` is binary, with labels `0` and `1`. `MulticlassLogistic` handles two or
more classes with any numeric labels. Each distinct label is a class. Its coefficients are
unique, but the intercepts are not, since adding one constant to every intercept leaves the
objective unchanged. The last class's intercept is pinned at 0.

## Model files

Every estimator here has `to_bytes` and `from_bytes`, in the format in
[docs/model-format.md](../../docs/model-format.md). Hyper-parameters are stored with the fitted
state, so a loaded model refits with the same settings.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Licence

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache-2.0](../../LICENSE-APACHE), at
your option.
