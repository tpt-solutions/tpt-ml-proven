# tpt-ml-proven-ensemble

Ensemble models for [tpt-ml-proven](https://github.com/tpt-solutions/tpt-ml-proven).

> Status: `StumpForest` is implemented. It is tested by properties, not against scikit-learn,
> since there is no scikit-learn model to match.

## StumpForest

A bagged ensemble of decision stumps (depth-one trees) for classification. Each stump is
fitted on a bootstrap sample, splits one feature at one threshold to minimise weighted Gini
impurity, and predicts a class on each side. The forest predicts by majority vote, with ties
going to the smaller label.

Contract: [`docs/contracts/stump-forest.tptc`](../../docs/contracts/stump-forest.tptc). Every
stump predicts one of the training classes.

```rust
use tpt_ml_proven_core::{DenseView, Predictor, SupervisedEstimator};
use tpt_ml_proven_ensemble::StumpForest;

let mut forest = StumpForest::<f64>::new(25).with_seed(7);
forest.fit(DenseView::new(&x, rows, cols).unwrap(), &y[..]).unwrap();
let labels = forest.predict(DenseView::new(&x, rows, cols).unwrap()).unwrap();
```

For regression trees and gradient boosting, see
[`tpt-ml-proven-trees`](../tpt-ml-proven-trees/README.md).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Licence

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache-2.0](../../LICENSE-APACHE), at
your option.
