// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! CART regression trees with exact greedy splits.
//!
//! Each split minimises the sum of squared errors of the two children, which is the
//! `friedman_mse` criterion of scikit-learn up to a constant. A split sends `x ≤ threshold`
//! left, and the threshold is the midpoint of two adjacent distinct values, as scikit-learn
//! chooses it. A leaf predicts the mean of its targets.

use tpt_ml_proven_core::error::Error;
use tpt_ml_proven_core::rng::SplitMix64;
use tpt_ml_proven_core::{ensures, requires};

/// One node of a tree, stored in an arena.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Node {
    /// A leaf predicting `value`, built from `count` training rows.
    Leaf { value: f64, count: usize },
    /// `x[feature] <= threshold` goes to node `left`, otherwise to `right`.
    Split {
        feature: usize,
        threshold: f64,
        left: usize,
        right: usize,
        count: usize,
    },
}

/// A fitted regression tree.
#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    pub(crate) nodes: Vec<Node>,
    pub(crate) n_features: usize,
    pub(crate) depth: usize,
}

/// Tree-growing limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Limits {
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    /// Features considered per split. `usize::MAX` means all of them.
    pub max_features: usize,
}

impl Tree {
    /// The number of features the tree was grown on.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// The tree's depth: 0 for a single leaf.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Number of leaves.
    pub fn n_leaves(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| matches!(n, Node::Leaf { .. }))
            .count()
    }

    /// Predicts the value for one row of `n_features` values.
    pub(crate) fn predict_row(&self, row: &[f64]) -> f64 {
        let mut at = 0;
        loop {
            match self.nodes[at] {
                Node::Leaf { value, .. } => return value,
                Node::Split {
                    feature,
                    threshold,
                    left,
                    right,
                    ..
                } => {
                    at = if row[feature] <= threshold {
                        left
                    } else {
                        right
                    }
                }
            }
        }
    }

    /// Sum of the leaf counts. Equals the training size if the leaves partition the data.
    pub(crate) fn leaf_count_total(&self) -> usize {
        self.nodes
            .iter()
            .map(|n| match n {
                Node::Leaf { count, .. } => *count,
                Node::Split { .. } => 0,
            })
            .sum()
    }
}

/// Fits a tree to `targets` over the row-major `data` (`rows × cols`). At each node only
/// `limits.max_features` features are candidates, drawn from `rng`. With every feature
/// allowed, `rng` is not used.
pub(crate) fn fit_tree(
    data: &[f64],
    rows: usize,
    cols: usize,
    targets: &[f64],
    limits: Limits,
    rng: &mut SplitMix64,
) -> Result<Tree, Error> {
    requires!(limits.max_depth >= 1, "tree.max_depth_positive");
    requires!(
        limits.min_samples_leaf >= 1,
        "tree.min_samples_leaf_positive"
    );
    requires!(rows >= 1, "tree.rows_positive");

    let mut nodes = Vec::new();
    let idx: Vec<usize> = (0..rows).collect();
    let mut depth = 0;
    let mut grower = Grower {
        data,
        cols,
        targets,
        limits,
        rng,
    };
    grower.grow(idx, 0, &mut nodes, &mut depth);
    let tree = Tree {
        nodes,
        n_features: cols,
        depth,
    };

    // Structural invariants of a grown tree.
    ensures!(
        tree.leaf_count_total() == rows,
        "tree.leaves_partition_rows"
    );
    ensures!(tree.depth() <= limits.max_depth, "tree.depth_bounded");
    Ok(tree)
}

/// The features a node may split on: all of them, or `max_features` drawn without repetition.
pub(crate) fn choose_features(
    cols: usize,
    max_features: usize,
    rng: &mut SplitMix64,
) -> Vec<usize> {
    let mut all: Vec<usize> = (0..cols).collect();
    if max_features >= cols {
        return all;
    }
    // Partial Fisher–Yates: the first `max_features` entries are a uniform sample.
    for i in 0..max_features {
        let j = i + rng.next_below(cols - i);
        all.swap(i, j);
    }
    all.truncate(max_features);
    all
}

/// `k` distinct row indices out of `0..rows`, drawn uniformly from `rng`.
pub(crate) fn sample_rows(rows: usize, k: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let mut all: Vec<usize> = (0..rows).collect();
    for i in 0..k {
        let j = i + rng.next_below(rows - i);
        all.swap(i, j);
    }
    all.truncate(k);
    all
}

/// The state shared by every node of one tree being grown.
struct Grower<'a> {
    data: &'a [f64],
    cols: usize,
    targets: &'a [f64],
    limits: Limits,
    rng: &'a mut SplitMix64,
}

impl Grower<'_> {
    /// Grows the subtree for `idx` and returns the index of its root in `nodes`.
    fn grow(
        &mut self,
        idx: Vec<usize>,
        depth: usize,
        nodes: &mut Vec<Node>,
        max_depth_seen: &mut usize,
    ) -> usize {
        let n = idx.len();
        let total: f64 = idx.iter().map(|&i| self.targets[i]).sum();
        let mean = total / n as f64;
        *max_depth_seen = (*max_depth_seen).max(depth);

        let sse: f64 = idx.iter().map(|&i| (self.targets[i] - mean).powi(2)).sum();
        let can_split =
            depth < self.limits.max_depth && n >= 2 * self.limits.min_samples_leaf && sse > 1e-12;
        let best = if can_split {
            let features = choose_features(self.cols, self.limits.max_features, self.rng);
            self.best_split(&idx, total, &features)
        } else {
            None
        };

        let Some(split) = best else {
            let at = nodes.len();
            nodes.push(Node::Leaf {
                value: mean,
                count: n,
            });
            return at;
        };

        let (data, cols) = (self.data, self.cols);
        let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = idx
            .into_iter()
            .partition(|&i| data[i * cols + split.feature] <= split.threshold);
        // Reserve this node's slot, then grow the children and fill the slot.
        let at = nodes.len();
        nodes.push(Node::Leaf {
            value: mean,
            count: n,
        });
        let left = self.grow(left_idx, depth + 1, nodes, max_depth_seen);
        let right = self.grow(right_idx, depth + 1, nodes, max_depth_seen);
        nodes[at] = Node::Split {
            feature: split.feature,
            threshold: split.threshold,
            left,
            right,
            count: n,
        };
        at
    }

    /// The split, among `features`, that most reduces squared error, or `None` if no valid
    /// split reduces it.
    fn best_split(&self, idx: &[usize], total: f64, features: &[usize]) -> Option<Split> {
        let (data, cols, targets) = (self.data, self.cols, self.targets);
        let min_leaf = self.limits.min_samples_leaf;
        let n = idx.len();
        // Maximising sl²/nl + sr²/nr is the same as minimising the children's squared error.
        let mut best_score = f64::NEG_INFINITY;
        let mut best: Option<Split> = None;
        for &feature in features {
            let mut order = idx.to_vec();
            order.sort_by(|&a, &b| data[a * cols + feature].total_cmp(&data[b * cols + feature]));
            let mut sum_left = 0.0;
            for k in 0..n - 1 {
                let i = order[k];
                sum_left += targets[i];
                let (nl, nr) = (k + 1, n - k - 1);
                let v = data[i * cols + feature];
                let next = data[order[k + 1] * cols + feature];
                if v == next || nl < min_leaf || nr < min_leaf {
                    continue;
                }
                let sum_right = total - sum_left;
                let score = sum_left * sum_left / nl as f64 + sum_right * sum_right / nr as f64;
                if score > best_score {
                    best_score = score;
                    // The midpoint, as scikit-learn chooses it; fall back to v if rounding hits next.
                    let mut threshold = v / 2.0 + next / 2.0;
                    if threshold == next {
                        threshold = v;
                    }
                    best = Some(Split { feature, threshold });
                }
            }
        }
        // Accept only a split that reduces error, i.e. beats the unsplit node.
        let parent = total * total / n as f64;
        best.filter(|_| best_score > parent + 1e-12 * (1.0 + parent.abs()))
    }
}

struct Split {
    feature: usize,
    threshold: f64,
}
