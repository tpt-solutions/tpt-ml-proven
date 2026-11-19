// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! CART classification trees with exact greedy splits.
//!
//! A split minimises the weighted impurity of its two children: `n·gini` or `n·entropy`,
//! the same quantity scikit-learn minimises up to the normalisation by `n`. Thresholds are
//! midpoints of adjacent distinct values, as in the regression tree. A leaf predicts its
//! majority class, with ties going to the smaller label. Labels are numbers; the classes
//! are the distinct labels seen in training.

use tpt_ml_proven_core::error::{Error, ShapeError};
use tpt_ml_proven_core::{
    ensures, requires, ArrayView, Predictor, SupervisedEstimator, TargetView,
};

use tpt_ml_proven_core::rng::SplitMix64;

use crate::tree::{choose_features, grow_best_first, partition_rows, Limits, Node, Split, Tree};

/// How a node's impurity is measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Criterion {
    /// Gini impurity, `1 − Σ p²`. scikit-learn's default.
    Gini,
    /// Shannon entropy, `−Σ p log p`, in nats.
    Entropy,
}

impl Criterion {
    /// `n` times the impurity of a node with the given class counts. Summing this over the two
    /// children of a split gives the quantity a split minimises.
    fn weighted_impurity(self, counts: &[usize], n: usize) -> f64 {
        if n == 0 {
            return 0.0;
        }
        let nf = n as f64;
        match self {
            // n · (1 − Σ (c/n)²) = n − Σ c²/n
            Criterion::Gini => nf - counts.iter().map(|&c| (c * c) as f64).sum::<f64>() / nf,
            // n · (−Σ (c/n) ln(c/n)) = Σ c · ln(n/c)
            Criterion::Entropy => counts
                .iter()
                .filter(|&&c| c > 0)
                .map(|&c| c as f64 * (nf / c as f64).ln())
                .sum(),
        }
    }
}

/// A CART classification tree.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionTreeClassifier {
    pub(crate) max_depth: usize,
    pub(crate) min_samples_leaf: usize,
    pub(crate) min_samples_split: usize,
    pub(crate) max_leaf_nodes: Option<usize>,
    pub(crate) criterion: Criterion,
    pub(crate) fitted: Option<Fitted>,
}

/// The learned part: the tree and the sorted class labels its leaves index into.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Fitted {
    pub(crate) tree: Tree,
    pub(crate) classes: Vec<f64>,
}

impl DecisionTreeClassifier {
    /// An unfitted tree of at most `max_depth` levels, using Gini impurity, with at least one
    /// row per leaf.
    pub fn new(max_depth: usize) -> Self {
        Self {
            max_depth,
            min_samples_leaf: 1,
            min_samples_split: 2,
            max_leaf_nodes: None,
            criterion: Criterion::Gini,
            fitted: None,
        }
    }

    /// Sets the minimum number of training rows in each leaf.
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Sets the number of training rows a node needs before it may be split (default 2).
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Limits the tree to this many leaves, grown best-first, or removes the limit with `None`.
    pub fn with_max_leaf_nodes(mut self, max_leaf_nodes: Option<usize>) -> Self {
        self.max_leaf_nodes = max_leaf_nodes;
        self
    }

    /// Chooses the impurity measure.
    pub fn with_criterion(mut self, criterion: Criterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// The distinct class labels seen in training, ascending, or `None` before `fit`.
    pub fn classes(&self) -> Option<&[f64]> {
        self.fitted.as_ref().map(|f| f.classes.as_slice())
    }

    /// Number of leaves in the fitted tree, or `None` before `fit`.
    pub fn n_leaves(&self) -> Option<usize> {
        self.fitted.as_ref().map(|f| f.tree.n_leaves())
    }

    /// Depth of the fitted tree, or `None` before `fit`.
    pub fn depth(&self) -> Option<usize> {
        self.fitted.as_ref().map(|f| f.tree.depth())
    }

    /// Each feature's share of the total impurity reduction, summing to one. `None` before `fit`,
    /// and for a model loaded from a file written before importances were stored.
    pub fn feature_importances(&self) -> Option<Vec<f64>> {
        self.fitted
            .as_ref()
            .and_then(|f| f.tree.feature_importances())
    }

    /// Fraction of rows of `x` whose predicted label equals the label in `y`.
    pub fn score<X: ArrayView<Elem = f64>>(&self, x: X, y: &[f64]) -> Result<f64, Error> {
        let pred = Predictor::predict(self, x)?;
        tpt_ml_proven_core::metrics::accuracy(y, &pred)
    }
}

impl Default for DecisionTreeClassifier {
    /// Depth 3, Gini impurity, at least one row per leaf.
    fn default() -> Self {
        Self::new(3)
    }
}

impl SupervisedEstimator for DecisionTreeClassifier {
    type Float = f64;
    type FitError = Error;

    fn fit<X, Y>(&mut self, x: X, y: Y) -> Result<(), Error>
    where
        X: ArrayView<Elem = f64>,
        Y: TargetView<Elem = f64>,
    {
        self.fitted = None;
        check_params(
            self.max_depth,
            self.min_samples_leaf,
            self.min_samples_split,
            self.max_leaf_nodes,
        )?;
        let (rows, cols) = (x.rows(), x.cols());
        if rows == 0 || cols == 0 {
            return Err(Error::EmptyInput);
        }
        if y.len() != rows {
            return Err(Error::Shape(ShapeError {
                rows,
                cols,
                reason: "target length differs from the number of rows",
            }));
        }
        if !x.all_finite() {
            return Err(Error::NonFinite { what: "X" });
        }
        let labels: Vec<f64> = (0..rows).map(|i| y.get(i)).collect();
        if labels.iter().any(|v| !v.is_finite()) {
            return Err(Error::NonFinite { what: "y" });
        }
        requires!(rows >= 1, "classifier.rows_positive");
        requires!(self.max_depth >= 1, "classifier.max_depth_positive");
        requires!(
            self.min_samples_leaf >= 1,
            "classifier.min_samples_leaf_positive"
        );

        let mut classes = labels.clone();
        classes.sort_by(f64::total_cmp);
        classes.dedup();
        let class_of: Vec<usize> = labels
            .iter()
            .map(|v| {
                classes
                    .binary_search_by(|c| c.total_cmp(v))
                    .expect("every label is one of the classes")
            })
            .collect();

        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(x.get(r, c));
            }
        }

        let limits = Limits {
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            min_samples_split: self.min_samples_split,
            max_leaf_nodes: self.max_leaf_nodes.unwrap_or(usize::MAX),
            max_features: usize::MAX,
        };
        let tree = grow_classification_tree(
            &data,
            cols,
            &class_of,
            &classes,
            limits,
            self.criterion,
            &mut SplitMix64::new(0),
        );

        ensures!(
            tree.leaf_count_total() == rows,
            "classifier.leaves_partition_rows"
        );
        ensures!(tree.depth() <= self.max_depth, "classifier.depth_bounded");
        ensures!(
            tree.nodes.iter().all(|n| match n {
                Node::Leaf { value, .. } => classes.contains(value),
                Node::Split { .. } => true,
            }),
            "classifier.predicts_training_class"
        );
        self.fitted = Some(Fitted { tree, classes });
        Ok(())
    }
}

impl Predictor for DecisionTreeClassifier {
    type Float = f64;
    type Output = Vec<f64>;

    /// The class label of each row's leaf.
    fn predict<X>(&self, x: X) -> Result<Vec<f64>, Error>
    where
        X: ArrayView<Elem = f64>,
    {
        let fitted = self.fitted.as_ref().ok_or(Error::NotFitted)?;
        if x.cols() != fitted.tree.n_features {
            return Err(Error::Shape(ShapeError {
                rows: x.rows(),
                cols: x.cols(),
                reason: "feature count differs from the fitted model",
            }));
        }
        if !x.all_finite() {
            return Err(Error::NonFinite { what: "X" });
        }
        let mut row = vec![0.0; x.cols()];
        let mut out = Vec::with_capacity(x.rows());
        for r in 0..x.rows() {
            for (c, slot) in row.iter_mut().enumerate() {
                *slot = x.get(r, c);
            }
            out.push(fitted.tree.predict_row(&row));
        }
        Ok(out)
    }
}

/// Checks the settings a classifier is fitted with. Shared with the file reader.
pub(crate) fn check_params(
    max_depth: usize,
    min_samples_leaf: usize,
    min_samples_split: usize,
    max_leaf_nodes: Option<usize>,
) -> Result<(), Error> {
    if max_depth == 0 {
        return Err(Error::InvalidParameter {
            name: "max_depth",
            reason: "must be at least 1",
        });
    }
    if min_samples_leaf == 0 {
        return Err(Error::InvalidParameter {
            name: "min_samples_leaf",
            reason: "must be at least 1",
        });
    }
    if min_samples_split < 2 {
        return Err(Error::InvalidParameter {
            name: "min_samples_split",
            reason: "must be at least 2",
        });
    }
    if max_leaf_nodes.is_some_and(|k| k < 2) {
        return Err(Error::InvalidParameter {
            name: "max_leaf_nodes",
            reason: "must be at least 2",
        });
    }
    Ok(())
}

/// Everything the recursion needs that does not change from node to node.
struct Grow<'a> {
    data: &'a [f64],
    cols: usize,
    class_of: &'a [usize],
    classes: &'a [f64],
    n_classes: usize,
    limits: Limits,
    criterion: Criterion,
}

fn class_counts(ctx: &Grow<'_>, idx: &[usize]) -> Vec<usize> {
    let mut counts = vec![0usize; ctx.n_classes];
    for &i in idx {
        counts[ctx.class_of[i]] += 1;
    }
    counts
}

/// Index of the largest count; ties go to the smallest index, i.e. the smaller label.
fn majority(counts: &[usize]) -> usize {
    let mut best = 0;
    for (k, &c) in counts.iter().enumerate() {
        if c > counts[best] {
            best = k;
        }
    }
    best
}

/// Grows one classification tree over `data` (`rows × cols`), whose rows have class indices
/// `class_of` into the sorted `classes`. At each node only `limits.max_features` features are
/// candidates, drawn from `rng`. Grows best-first when `limits.max_leaf_nodes` is finite.
pub(crate) fn grow_classification_tree(
    data: &[f64],
    cols: usize,
    class_of: &[usize],
    classes: &[f64],
    limits: Limits,
    criterion: Criterion,
    rng: &mut SplitMix64,
) -> Tree {
    let ctx = Grow {
        data,
        cols,
        class_of,
        classes,
        n_classes: classes.len(),
        limits,
        criterion,
    };
    let rows: Vec<usize> = (0..class_of.len()).collect();
    let mut importances = vec![0.0; cols];
    let (nodes, depth) = if limits.max_leaf_nodes == usize::MAX {
        let mut nodes = Vec::new();
        let mut depth = 0;
        grow(&ctx, rng, rows, 0, &mut nodes, &mut depth, &mut importances);
        (nodes, depth)
    } else {
        grow_best_first(
            rows,
            data,
            cols,
            limits.max_leaf_nodes,
            &mut importances,
            |idx, depth| evaluate(&ctx, rng, idx, depth),
        )
    };
    Tree {
        nodes,
        n_features: cols,
        depth,
        importances,
    }
}

/// The leaf record for `idx` and, if the node may be split, its best split.
fn evaluate(
    ctx: &Grow<'_>,
    rng: &mut SplitMix64,
    idx: &[usize],
    depth: usize,
) -> (Node, Option<Split>) {
    let n = idx.len();
    let counts = class_counts(ctx, idx);
    let leaf_class = majority(&counts);
    let leaf = Node::Leaf {
        value: ctx.classes[leaf_class],
        count: n,
    };
    let parent = ctx.criterion.weighted_impurity(&counts, n);
    let can_split = depth < ctx.limits.max_depth
        && n >= ctx.limits.split_threshold()
        && parent > 1e-12 * n as f64;
    let split = if can_split {
        let features = choose_features(ctx.cols, ctx.limits.max_features, rng);
        best_split(ctx, idx, parent, &features)
    } else {
        None
    };
    (leaf, split)
}

/// Grows the subtree for `idx` depth-first and returns the index of its root in `nodes`.
fn grow(
    ctx: &Grow<'_>,
    rng: &mut SplitMix64,
    idx: Vec<usize>,
    depth: usize,
    nodes: &mut Vec<Node>,
    max_depth_seen: &mut usize,
    importances: &mut [f64],
) -> usize {
    let n = idx.len();
    *max_depth_seen = (*max_depth_seen).max(depth);
    let (leaf, split) = evaluate(ctx, rng, &idx, depth);
    let Some(split) = split else {
        let at = nodes.len();
        nodes.push(leaf);
        return at;
    };
    importances[split.feature] += split.gain;

    let (left_idx, right_idx) =
        partition_rows(idx, ctx.data, ctx.cols, split.feature, split.threshold);
    let at = nodes.len();
    nodes.push(leaf);
    let left = grow(
        ctx,
        rng,
        left_idx,
        depth + 1,
        nodes,
        max_depth_seen,
        importances,
    );
    let right = grow(
        ctx,
        rng,
        right_idx,
        depth + 1,
        nodes,
        max_depth_seen,
        importances,
    );
    nodes[at] = Node::Split {
        feature: split.feature,
        threshold: split.threshold,
        left,
        right,
        count: n,
    };
    at
}

/// The split whose two children have the least weighted impurity, provided that does not
/// exceed the unsplit node's. Its gain is the reduction in weighted impurity. `None` if no valid
/// split exists.
fn best_split(ctx: &Grow<'_>, idx: &[usize], parent: f64, features: &[usize]) -> Option<Split> {
    let n = idx.len();
    let total = class_counts(ctx, idx);
    let mut best_score = f64::INFINITY;
    let mut best: Option<(usize, f64)> = None;
    for &feature in features {
        let mut order = idx.to_vec();
        order.sort_by(|&a, &b| {
            ctx.data[a * ctx.cols + feature].total_cmp(&ctx.data[b * ctx.cols + feature])
        });
        let mut left = vec![0usize; ctx.n_classes];
        for k in 0..n - 1 {
            let i = order[k];
            left[ctx.class_of[i]] += 1;
            let (nl, nr) = (k + 1, n - k - 1);
            let v = ctx.data[i * ctx.cols + feature];
            let next = ctx.data[order[k + 1] * ctx.cols + feature];
            if v == next || nl < ctx.limits.min_samples_leaf || nr < ctx.limits.min_samples_leaf {
                continue;
            }
            let right: Vec<usize> = total.iter().zip(&left).map(|(t, l)| t - l).collect();
            let score = ctx.criterion.weighted_impurity(&left, nl)
                + ctx.criterion.weighted_impurity(&right, nr);
            if score < best_score {
                best_score = score;
                // The midpoint, as scikit-learn chooses it; fall back to v if rounding hits next.
                let mut threshold = v / 2.0 + next / 2.0;
                if threshold == next {
                    threshold = v;
                }
                best = Some((feature, threshold));
            }
        }
    }
    // As in scikit-learn with min_impurity_decrease = 0, a zero-gain split is still taken, so
    // that XOR-like data reaches depth two. An impure node always gets its best valid split.
    let (feature, threshold) = best?;
    if best_score <= parent + 1e-12 * (1.0 + parent.abs()) {
        Some(Split {
            feature,
            threshold,
            gain: (parent - best_score).max(0.0),
        })
    } else {
        None
    }
}

// ---------------------------------------------------------------- settings

impl DecisionTreeClassifier {
    /// Maximum depth the tree was configured with.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Minimum rows per leaf the tree was configured with.
    pub fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    /// Minimum rows a node needs before it may be split.
    pub fn min_samples_split(&self) -> usize {
        self.min_samples_split
    }

    /// The leaf limit the tree was configured with, or `None` for no limit.
    pub fn max_leaf_nodes(&self) -> Option<usize> {
        self.max_leaf_nodes
    }

    /// Impurity measure the tree was configured with.
    pub fn criterion(&self) -> Criterion {
        self.criterion
    }
}

impl DecisionTreeClassifier {
    /// The number of features the fitted tree was grown on, or `None` before `fit`.
    pub fn n_features(&self) -> Option<usize> {
        self.fitted.as_ref().map(|f| f.tree.n_features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ml_proven_core::DenseView;

    fn fit(model: &mut DecisionTreeClassifier, x: &[f64], rows: usize, cols: usize, y: &[f64]) {
        SupervisedEstimator::fit(model, DenseView::new(x, rows, cols).unwrap(), y).unwrap();
    }

    #[test]
    fn a_separable_problem_is_learned_exactly() {
        // Class 0 below 2, class 1 above, on one feature; a second feature is noise.
        let x: Vec<f64> = (0..20)
            .flat_map(|i| [i as f64, ((i * 7) % 5) as f64])
            .collect();
        let y: Vec<f64> = (0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }).collect();
        let mut t = DecisionTreeClassifier::new(3);
        fit(&mut t, &x, 20, 2, &y);
        let pred = Predictor::predict(&t, DenseView::new(&x, 20, 2).unwrap()).unwrap();
        assert_eq!(pred, y);
        assert_eq!(t.classes().unwrap(), &[0.0, 1.0]);
    }

    #[test]
    fn entropy_also_separates_the_classes() {
        let x: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..12).map(|i| if i < 6 { -1.0 } else { 1.0 }).collect();
        let mut t = DecisionTreeClassifier::new(2).with_criterion(Criterion::Entropy);
        fit(&mut t, &x, 12, 1, &y);
        assert_eq!(
            t.score(DenseView::new(&x, 12, 1).unwrap(), &y).unwrap(),
            1.0
        );
    }

    #[test]
    fn xor_needs_depth_two() {
        let x = [0.0_f64, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let y = [0.0_f64, 1.0, 1.0, 0.0];
        let mut stump = DecisionTreeClassifier::new(1);
        fit(&mut stump, &x, 4, 2, &y);
        assert!(stump.score(DenseView::new(&x, 4, 2).unwrap(), &y).unwrap() < 1.0);
        let mut deep = DecisionTreeClassifier::new(2);
        fit(&mut deep, &x, 4, 2, &y);
        assert_eq!(
            deep.score(DenseView::new(&x, 4, 2).unwrap(), &y).unwrap(),
            1.0
        );
    }

    #[test]
    fn a_single_class_gives_one_leaf() {
        let x = [1.0_f64, 2.0, 3.0];
        let y = [4.0_f64, 4.0, 4.0];
        let mut t = DecisionTreeClassifier::new(5);
        fit(&mut t, &x, 3, 1, &y);
        assert_eq!(t.n_leaves(), Some(1));
        assert_eq!(
            Predictor::predict(&t, DenseView::new(&x, 3, 1).unwrap()).unwrap(),
            vec![4.0; 3]
        );
    }

    #[test]
    fn min_samples_leaf_limits_the_split() {
        let x: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let y = [0.0_f64, 0.0, 0.0, 0.0, 0.0, 1.0];
        let mut t = DecisionTreeClassifier::new(4).with_min_samples_leaf(2);
        fit(&mut t, &x, 6, 1, &y);
        // The lone class-1 row cannot get a leaf of its own, so it is absorbed.
        assert_eq!(
            t.score(DenseView::new(&x, 6, 1).unwrap(), &y).unwrap(),
            5.0 / 6.0
        );
    }

    #[test]
    fn unfitted_and_bad_parameters_are_errors() {
        let x = [0.0_f64, 1.0];
        let t = DecisionTreeClassifier::new(2);
        assert_eq!(
            Predictor::predict(&t, DenseView::new(&x, 2, 1).unwrap()).unwrap_err(),
            Error::NotFitted
        );
        let mut bad = DecisionTreeClassifier::new(0);
        assert!(matches!(
            SupervisedEstimator::fit(&mut bad, DenseView::new(&x, 2, 1).unwrap(), &[0.0, 1.0][..]),
            Err(Error::InvalidParameter {
                name: "max_depth",
                ..
            })
        ));
    }

    #[test]
    fn min_samples_split_blocks_small_nodes() {
        // Six rows, one split needed. A split threshold of 7 means the root may not split.
        let x: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let y = [0.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut t = DecisionTreeClassifier::new(4).with_min_samples_split(7);
        fit(&mut t, &x, 6, 1, &y);
        assert_eq!(t.n_leaves(), Some(1));
        let mut ok = DecisionTreeClassifier::new(4).with_min_samples_split(6);
        fit(&mut ok, &x, 6, 1, &y);
        assert_eq!(ok.n_leaves(), Some(2));
    }

    #[test]
    fn max_leaf_nodes_caps_the_leaves_and_keeps_the_best_split() {
        // Three classes in three bands of three rows on one feature. Both cuts give the same
        // gain at the root, so the first (at 2.5) is taken. Each further leaf is then the best
        // remaining split, which is how the tree reaches three perfectly classified leaves.
        let x: Vec<f64> = (0..9).map(|i| i as f64).collect();
        let y = [0.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let mut t = DecisionTreeClassifier::new(10).with_max_leaf_nodes(Some(2));
        fit(&mut t, &x, 9, 1, &y);
        assert_eq!(t.n_leaves(), Some(2));
        let mut more = DecisionTreeClassifier::new(10).with_max_leaf_nodes(Some(3));
        fit(&mut more, &x, 9, 1, &y);
        assert_eq!(more.n_leaves(), Some(3));
        assert_eq!(
            more.score(DenseView::new(&x, 9, 1).unwrap(), &y).unwrap(),
            1.0
        );
    }

    #[test]
    fn importances_sum_to_one_and_ignore_the_noise_feature() {
        // Feature 0 decides the class; feature 1 is constant, so it can never be split on.
        let x: Vec<f64> = (0..12).flat_map(|i| [i as f64, 3.0]).collect();
        let y: Vec<f64> = (0..12).map(|i| if i < 6 { 0.0 } else { 1.0 }).collect();
        let mut t = DecisionTreeClassifier::new(3);
        fit(&mut t, &x, 12, 2, &y);
        let imp = t.feature_importances().unwrap();
        assert!((imp.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(imp, vec![1.0, 0.0]);
    }

    #[test]
    fn a_tree_that_never_splits_has_zero_importances() {
        let x = [1.0_f64, 2.0, 3.0];
        let y = [5.0_f64, 5.0, 5.0];
        let mut t = DecisionTreeClassifier::new(3);
        fit(&mut t, &x, 3, 1, &y);
        assert_eq!(t.feature_importances().unwrap(), vec![0.0]);
        assert_eq!(DecisionTreeClassifier::new(3).feature_importances(), None);
    }

    #[test]
    fn invalid_split_and_leaf_limits_are_rejected() {
        let x = [0.0_f64, 1.0];
        let y = [0.0_f64, 1.0];
        let mut split = DecisionTreeClassifier::new(2).with_min_samples_split(1);
        assert!(matches!(
            SupervisedEstimator::fit(&mut split, DenseView::new(&x, 2, 1).unwrap(), &y[..]),
            Err(Error::InvalidParameter {
                name: "min_samples_split",
                ..
            })
        ));
        let mut leaves = DecisionTreeClassifier::new(2).with_max_leaf_nodes(Some(1));
        assert!(matches!(
            SupervisedEstimator::fit(&mut leaves, DenseView::new(&x, 2, 1).unwrap(), &y[..]),
            Err(Error::InvalidParameter {
                name: "max_leaf_nodes",
                ..
            })
        ));
    }
}
