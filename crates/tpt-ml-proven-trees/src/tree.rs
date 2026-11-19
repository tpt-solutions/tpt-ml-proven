// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! CART regression trees with exact greedy splits.
//!
//! Each split minimises the sum of squared errors of the two children, which is the
//! `friedman_mse` criterion of scikit-learn up to a constant. A split sends `x ≤ threshold`
//! left, and the threshold is the midpoint of two adjacent distinct values, as scikit-learn
//! chooses it. A leaf predicts the mean of its targets.
//!
//! Trees grow depth-first by default. With a `max_leaf_nodes` limit they grow best-first, as
//! scikit-learn does: the open node whose split reduces the error most is split next, until
//! the limit is reached.

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
    /// Sum, over the splits on each feature, of the reduction in impurity they made. Empty when
    /// the tree was loaded from a file that did not store it.
    pub(crate) importances: Vec<f64>,
}

/// Tree-growing limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Limits {
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    /// A node needs at least this many rows to be split.
    pub min_samples_split: usize,
    /// Growth stops at this many leaves, growing best-first. `usize::MAX` means no limit.
    pub max_leaf_nodes: usize,
    /// Features considered per split. `usize::MAX` means all of them.
    pub max_features: usize,
}

impl Limits {
    /// The fewest rows a node needs before it may be split.
    pub(crate) fn split_threshold(&self) -> usize {
        (2 * self.min_samples_leaf).max(self.min_samples_split)
    }
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

    /// Each feature's share of the total impurity reduction, summing to one (all zeros for a
    /// tree that never split). `None` when the tree was loaded without importances.
    pub fn feature_importances(&self) -> Option<Vec<f64>> {
        if self.importances.is_empty() {
            return None;
        }
        let total: f64 = self.importances.iter().sum();
        if total > 0.0 {
            Some(self.importances.iter().map(|g| g / total).collect())
        } else {
            Some(vec![0.0; self.importances.len()])
        }
    }

    /// The tree without its importances. Trees inside an ensemble do not store them.
    pub(crate) fn without_importances(mut self) -> Self {
        self.importances.clear();
        self
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

    let idx: Vec<usize> = (0..rows).collect();
    let mut grower = Grower {
        data,
        cols,
        targets,
        limits,
        rng,
        importances: vec![0.0; cols],
    };
    let (nodes, depth) = if limits.max_leaf_nodes == usize::MAX {
        let mut nodes = Vec::new();
        let mut depth = 0;
        grower.grow(idx, 0, &mut nodes, &mut depth);
        (nodes, depth)
    } else {
        // The importances are moved out so the closure can borrow the grower.
        let mut importances = std::mem::take(&mut grower.importances);
        let built = grow_best_first(
            idx,
            data,
            cols,
            limits.max_leaf_nodes,
            &mut importances,
            |rows, depth| grower.evaluate(rows, depth),
        );
        grower.importances = importances;
        built
    };
    let tree = Tree {
        nodes,
        n_features: cols,
        depth,
        importances: grower.importances,
    };

    // Structural invariants of a grown tree.
    ensures!(
        tree.leaf_count_total() == rows,
        "tree.leaves_partition_rows"
    );
    ensures!(tree.depth() <= limits.max_depth, "tree.depth_bounded");
    Ok(tree)
}

/// Split `idx` by `x[feature] <= threshold`, keeping the order of the rows.
pub(crate) fn partition_rows(
    idx: Vec<usize>,
    data: &[f64],
    cols: usize,
    feature: usize,
    threshold: f64,
) -> (Vec<usize>, Vec<usize>) {
    idx.into_iter()
        .partition(|&i| data[i * cols + feature] <= threshold)
}

/// A node whose rows are not yet split, waiting in the best-first frontier.
struct Open {
    slot: usize,
    idx: Vec<usize>,
    depth: usize,
    split: Split,
}

/// Grows a tree best-first. `evaluate` gives a node's leaf record and, when the node may be
/// split, its best split. Each step splits the open node with the largest gain, earliest in
/// the frontier on ties, until the tree has `max_leaves` leaves or nothing is left to split.
/// Each split's gain is added to `importances`. Returns the nodes and the depth reached.
pub(crate) fn grow_best_first<F>(
    rows: Vec<usize>,
    data: &[f64],
    cols: usize,
    max_leaves: usize,
    importances: &mut [f64],
    mut evaluate: F,
) -> (Vec<Node>, usize)
where
    F: FnMut(&[usize], usize) -> (Node, Option<Split>),
{
    let mut nodes = Vec::new();
    let mut depth_reached = 0;
    let (root, root_split) = evaluate(&rows, 0);
    nodes.push(root);
    let mut open: Vec<Open> = Vec::new();
    if let Some(split) = root_split {
        open.push(Open {
            slot: 0,
            idx: rows,
            depth: 0,
            split,
        });
    }
    let mut leaves = 1;
    while leaves < max_leaves {
        let mut pick: Option<usize> = None;
        for (k, o) in open.iter().enumerate() {
            if pick.is_none_or(|p| o.split.gain > open[p].split.gain) {
                pick = Some(k);
            }
        }
        let Some(k) = pick else { break };
        let o = open.remove(k);
        let Split {
            feature,
            threshold,
            gain,
        } = o.split;
        importances[feature] += gain;
        let count = o.idx.len();
        let (left_rows, right_rows) = partition_rows(o.idx, data, cols, feature, threshold);
        let child_depth = o.depth + 1;
        let (left_node, left_split) = evaluate(&left_rows, child_depth);
        let (right_node, right_split) = evaluate(&right_rows, child_depth);
        let left = nodes.len();
        nodes.push(left_node);
        let right = nodes.len();
        nodes.push(right_node);
        nodes[o.slot] = Node::Split {
            feature,
            threshold,
            left,
            right,
            count,
        };
        depth_reached = depth_reached.max(child_depth);
        if let Some(split) = left_split {
            open.push(Open {
                slot: left,
                idx: left_rows,
                depth: child_depth,
                split,
            });
        }
        if let Some(split) = right_split {
            open.push(Open {
                slot: right,
                idx: right_rows,
                depth: child_depth,
                split,
            });
        }
        leaves += 1;
    }
    (nodes, depth_reached)
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
    importances: Vec<f64>,
}

impl Grower<'_> {
    /// The leaf record for `idx` and, if the node may be split, its best split.
    fn evaluate(&mut self, idx: &[usize], depth: usize) -> (Node, Option<Split>) {
        let n = idx.len();
        let total: f64 = idx.iter().map(|&i| self.targets[i]).sum();
        let mean = total / n as f64;
        let sse: f64 = idx.iter().map(|&i| (self.targets[i] - mean).powi(2)).sum();
        let leaf = Node::Leaf {
            value: mean,
            count: n,
        };
        let can_split =
            depth < self.limits.max_depth && n >= self.limits.split_threshold() && sse > 1e-12;
        let split = if can_split {
            let features = choose_features(self.cols, self.limits.max_features, self.rng);
            self.best_split(idx, total, &features)
        } else {
            None
        };
        (leaf, split)
    }

    /// Grows the subtree for `idx` depth-first and returns the index of its root in `nodes`.
    fn grow(
        &mut self,
        idx: Vec<usize>,
        depth: usize,
        nodes: &mut Vec<Node>,
        max_depth_seen: &mut usize,
    ) -> usize {
        let n = idx.len();
        *max_depth_seen = (*max_depth_seen).max(depth);
        let (leaf, split) = self.evaluate(&idx, depth);
        let Some(split) = split else {
            let at = nodes.len();
            nodes.push(leaf);
            return at;
        };
        self.importances[split.feature] += split.gain;

        let (left_idx, right_idx) =
            partition_rows(idx, self.data, self.cols, split.feature, split.threshold);
        // Reserve this node's slot, then grow the children and fill the slot.
        let at = nodes.len();
        nodes.push(leaf);
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
    /// split reduces it. Its gain is the reduction in squared error.
    fn best_split(&self, idx: &[usize], total: f64, features: &[usize]) -> Option<Split> {
        let (data, cols, targets) = (self.data, self.cols, self.targets);
        let min_leaf = self.limits.min_samples_leaf;
        let n = idx.len();
        // Maximising sl²/nl + sr²/nr is the same as minimising the children's squared error.
        let mut best_score = f64::NEG_INFINITY;
        let mut best: Option<(usize, f64)> = None;
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
                    best = Some((feature, threshold));
                }
            }
        }
        // Accept only a split that reduces error, i.e. beats the unsplit node.
        let parent = total * total / n as f64;
        let (feature, threshold) = best?;
        if best_score > parent + 1e-12 * (1.0 + parent.abs()) {
            Some(Split {
                feature,
                threshold,
                gain: best_score - parent,
            })
        } else {
            None
        }
    }
}

/// The split a node takes: the feature, the threshold, and the reduction in impurity it gives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Split {
    pub feature: usize,
    pub threshold: f64,
    pub gain: f64,
}
