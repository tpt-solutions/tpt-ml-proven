// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! HDBSCAN: hierarchical density-based clustering, following scikit-learn's algorithm.
//!
//! The steps are: core distances (distance to the `min_samples`-th nearest sample, counting
//! the sample itself), the mutual-reachability graph, its minimum spanning tree (Prim), the
//! single-linkage tree, condensation with `min_cluster_size`, stability of each cluster, and
//! excess-of-mass (EOM) selection of the flat clusters.
//!
//! Distances are Euclidean and the pairwise matrix is held in memory, so the cost is O(n²).
//! Labels are `0..k` for clusters, in ascending order of the selected tree nodes, and `-1`
//! for noise. Ties in distances are broken by index, so results can differ from
//! scikit-learn when distances tie exactly.

use std::collections::BTreeMap;

use tpt_ml_proven_core::error::Error;
use tpt_ml_proven_core::{ensures, requires, ArrayView, Float, UnsupervisedEstimator};

use crate::kmeans::{for_each_row, sparse_rows, sparse_sq_dist, sq_dist};

/// Label given to noise samples.
pub const NOISE: i64 = -1;

/// One merge of the single-linkage tree. Node ids: samples are `0..n`, merge `i` is `n + i`.
#[derive(Debug, Clone, Copy)]
struct Merge {
    left: usize,
    right: usize,
    distance: f64,
    size: usize,
}

/// One row of the condensed tree: `child` leaves `parent` at `lambda = 1 / distance`.
/// A child is a cluster (`size > 1`) or a single sample (`size == 1`).
#[derive(Debug, Clone, Copy)]
struct Condensed {
    parent: usize,
    child: usize,
    lambda: f64,
    size: usize,
}

/// HDBSCAN clustering with excess-of-mass cluster selection.
#[derive(Debug, Clone, PartialEq)]
pub struct Hdbscan<F: Float> {
    pub(crate) min_cluster_size: usize,
    pub(crate) min_samples: Option<usize>,
    pub(crate) labels: Option<Vec<i64>>,
    marker: core::marker::PhantomData<F>,
}

impl<F: Float> Hdbscan<F> {
    /// Unfitted HDBSCAN. `min_samples` defaults to `min_cluster_size`.
    pub fn new(min_cluster_size: usize) -> Self {
        Self {
            min_cluster_size,
            min_samples: None,
            labels: None,
            marker: core::marker::PhantomData,
        }
    }

    /// Sets `min_samples`, the neighbourhood size that defines core distances.
    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = Some(min_samples);
        self
    }

    /// Cluster labels from the last fit, `-1` for noise, or `None` before `fit`.
    pub fn labels(&self) -> Option<&[i64]> {
        self.labels.as_deref()
    }

    /// Number of clusters found by the last fit (noise excluded), or `None` before `fit`.
    pub fn n_clusters(&self) -> Option<usize> {
        self.labels
            .as_ref()
            .map(|l| l.iter().copied().max().map_or(0, |m| (m + 1) as usize))
    }
}

/// Euclidean distance matrix, row-major `n × n`.
fn distance_matrix(data: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in i + 1..n {
            let v = sq_dist(&data[i * d..(i + 1) * d], &data[j * d..(j + 1) * d]).sqrt();
            out[i * n + j] = v;
            out[j * n + i] = v;
        }
    }
    out
}

/// Euclidean distance matrix of sparse rows, row-major `n × n`. The matrix is dense by nature, but
/// the input is read from its non-zeros only.
fn distance_matrix_sparse<F: Float>(rows: &[(Vec<usize>, Vec<F>)]) -> Vec<f64> {
    let n = rows.len();
    let mut out = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in i + 1..n {
            let (ai, av) = &rows[i];
            let (bi, bv) = &rows[j];
            let v = sparse_sq_dist(ai, av, bi, bv).to_f64().sqrt();
            out[i * n + j] = v;
            out[j * n + i] = v;
        }
    }
    out
}

/// Prim's algorithm on the mutual-reachability graph. Returns `n − 1` edges `(a, b, weight)`.
fn mutual_reachability_mst(dist: &[f64], n: usize, min_samples: usize) -> Vec<(usize, usize, f64)> {
    // Core distance: the min_samples-th smallest distance in the row, self included.
    let core: Vec<f64> = (0..n)
        .map(|i| {
            let mut row: Vec<f64> = dist[i * n..(i + 1) * n].to_vec();
            row.sort_by(f64::total_cmp);
            row[min_samples - 1]
        })
        .collect();
    let reach = |i: usize, j: usize| dist[i * n + j].max(core[i]).max(core[j]);

    let mut in_tree = vec![false; n];
    let mut best = vec![f64::INFINITY; n];
    let mut best_from = vec![0usize; n];
    in_tree[0] = true;
    for (j, b) in best.iter_mut().enumerate().skip(1) {
        *b = reach(0, j);
    }
    let mut edges = Vec::with_capacity(n - 1);
    for _ in 1..n {
        let next = (0..n)
            .filter(|&j| !in_tree[j])
            .min_by(|&a, &b| best[a].total_cmp(&best[b]))
            .expect("a vertex remains outside the tree");
        in_tree[next] = true;
        edges.push((best_from[next], next, best[next]));
        for j in 0..n {
            if !in_tree[j] {
                let w = reach(next, j);
                if w < best[j] {
                    best[j] = w;
                    best_from[j] = next;
                }
            }
        }
    }
    edges
}

/// Single-linkage tree from MST edges, processed in ascending weight.
fn single_linkage(mut edges: Vec<(usize, usize, f64)>, n: usize) -> Vec<Merge> {
    edges.sort_by(|a, b| a.2.total_cmp(&b.2));
    // Union-find over samples; each set remembers its current tree node and size.
    let mut parent: Vec<usize> = (0..n).collect();
    let mut node_of: Vec<usize> = (0..n).collect();
    let mut size: Vec<usize> = vec![1; n];
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut tree = Vec::with_capacity(n.saturating_sub(1));
    for (i, &(a, b, w)) in edges.iter().enumerate() {
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        let merged = size[ra] + size[rb];
        tree.push(Merge {
            left: node_of[ra],
            right: node_of[rb],
            distance: w,
            size: merged,
        });
        parent[ra] = rb;
        size[rb] = merged;
        node_of[rb] = n + i;
    }
    tree
}

/// Breadth-first list of a node and all its descendants, parents before children.
fn bfs_from_tree(tree: &[Merge], n: usize, root: usize) -> Vec<usize> {
    let mut result = Vec::new();
    let mut frontier = vec![root];
    while !frontier.is_empty() {
        result.extend_from_slice(&frontier);
        frontier = frontier
            .iter()
            .filter(|&&node| node >= n)
            .flat_map(|&node| {
                let m = tree[node - n];
                [m.left, m.right]
            })
            .collect();
    }
    result
}

/// Condenses the single-linkage tree: splits where both sides have at least
/// `min_cluster_size` samples create clusters; otherwise the small side's samples fall out
/// of the cluster at the split's lambda.
fn condense(tree: &[Merge], n: usize, min_cluster_size: usize) -> Vec<Condensed> {
    let size_of = |id: usize| if id < n { 1 } else { tree[id - n].size };
    let root = 2 * tree.len();
    let mut relabel = vec![0usize; root + 1];
    relabel[root] = n;
    let mut next_label = n + 1;
    let mut ignore = vec![false; root + 1];
    let mut rows = Vec::new();
    let node_list = bfs_from_tree(tree, n, root);
    for &node in &node_list {
        if ignore[node] || node < n {
            continue;
        }
        let m = tree[node - n];
        let lambda = if m.distance > 0.0 {
            1.0 / m.distance
        } else {
            f64::INFINITY
        };
        let (lc, rc) = (size_of(m.left), size_of(m.right));
        let mut drop_samples = |rows: &mut Vec<Condensed>, sub_root: usize, label: usize| {
            for sub in bfs_from_tree(tree, n, sub_root) {
                ignore[sub] = true;
                if sub < n {
                    rows.push(Condensed {
                        parent: label,
                        child: sub,
                        lambda,
                        size: 1,
                    });
                }
            }
        };
        if lc >= min_cluster_size && rc >= min_cluster_size {
            relabel[m.left] = next_label;
            next_label += 1;
            rows.push(Condensed {
                parent: relabel[node],
                child: relabel[m.left],
                lambda,
                size: lc,
            });
            relabel[m.right] = next_label;
            next_label += 1;
            rows.push(Condensed {
                parent: relabel[node],
                child: relabel[m.right],
                lambda,
                size: rc,
            });
        } else if lc < min_cluster_size && rc < min_cluster_size {
            let label = relabel[node];
            drop_samples(&mut rows, m.left, label);
            drop_samples(&mut rows, m.right, label);
        } else if lc < min_cluster_size {
            relabel[m.right] = relabel[node];
            drop_samples(&mut rows, m.left, relabel[node]);
        } else {
            relabel[m.left] = relabel[node];
            drop_samples(&mut rows, m.right, relabel[node]);
        }
    }
    rows
}

/// Stability of every cluster: `Σ (λ − birth) · size` over the rows it parents.
fn stabilities(condensed: &[Condensed], n: usize) -> BTreeMap<usize, f64> {
    let max_id = condensed
        .iter()
        .map(|r| r.child.max(r.parent))
        .max()
        .unwrap_or(n);
    let mut births = vec![f64::NAN; max_id + 1];
    for r in condensed {
        let b = &mut births[r.child];
        if b.is_nan() || r.lambda < *b {
            *b = r.lambda;
        }
    }
    births[n] = 0.0;
    let mut stab: BTreeMap<usize, f64> = BTreeMap::new();
    for r in condensed {
        *stab.entry(r.parent).or_insert(0.0) += (r.lambda - births[r.parent]) * r.size as f64;
    }
    stab
}

/// Excess-of-mass selection: a cluster is kept unless its selected descendants have more
/// total stability. The root is not selected.
fn select_clusters(condensed: &[Condensed], n: usize) -> Vec<usize> {
    let mut stab = stabilities(condensed, n);
    let cluster_rows: Vec<&Condensed> = condensed.iter().filter(|r| r.size > 1).collect();
    let mut is_cluster: BTreeMap<usize, bool> = stab.keys().map(|&k| (k, true)).collect();
    let nodes: Vec<usize> = stab.keys().rev().copied().collect();
    for node in nodes {
        let children: Vec<usize> = cluster_rows
            .iter()
            .filter(|r| r.parent == node)
            .map(|r| r.child)
            .collect();
        let subtree: f64 = children
            .iter()
            .map(|c| stab.get(c).copied().unwrap_or(0.0))
            .sum();
        if subtree > stab[&node] {
            is_cluster.insert(node, false);
            stab.insert(node, subtree);
        } else {
            // Deselect every descendant cluster.
            let mut stack = children;
            while let Some(c) = stack.pop() {
                is_cluster.insert(c, false);
                stack.extend(
                    cluster_rows
                        .iter()
                        .filter(|r| r.parent == c)
                        .map(|r| r.child),
                );
            }
        }
    }
    is_cluster.insert(n, false);
    is_cluster
        .into_iter()
        .filter(|&(_, selected)| selected)
        .map(|(c, _)| c)
        .collect()
}

impl<F: Float> UnsupervisedEstimator for Hdbscan<F> {
    type Float = F;
    type FitError = Error;

    fn fit<X>(&mut self, x: X) -> Result<(), Error>
    where
        X: ArrayView<Elem = F>,
    {
        self.labels = None;
        let (rows, cols) = (x.rows(), x.cols());
        if rows == 0 || cols == 0 {
            return Err(Error::EmptyInput);
        }
        if self.min_cluster_size < 2 {
            return Err(Error::InvalidParameter {
                name: "min_cluster_size",
                reason: "must be at least 2",
            });
        }
        let min_samples = self.min_samples.unwrap_or(self.min_cluster_size);
        if min_samples == 0 || min_samples > rows {
            return Err(Error::InvalidParameter {
                name: "min_samples",
                reason: "must be between 1 and the number of samples",
            });
        }
        if !x.all_finite() {
            return Err(Error::NonFinite { what: "X" });
        }
        requires!(rows >= 1, "hdbscan.rows_positive");
        requires!(self.min_cluster_size >= 2, "hdbscan.min_cluster_size_min");
        requires!(min_samples >= 1, "hdbscan.min_samples_positive");
        if rows == 1 {
            self.labels = Some(vec![NOISE]);
            return Ok(());
        }
        let dist = if x.is_sparse() {
            distance_matrix_sparse(&sparse_rows(&x, rows))
        } else {
            let mut data: Vec<f64> = Vec::with_capacity(rows * cols);
            for_each_row(&x, |_, row| data.extend(row.iter().map(|v| v.to_f64())));
            distance_matrix(&data, rows, cols)
        };
        let mst = mutual_reachability_mst(&dist, rows, min_samples);
        let tree = single_linkage(mst, rows);
        let condensed = condense(&tree, rows, self.min_cluster_size);
        let selected = select_clusters(&condensed, rows);

        // A sample's label is its nearest selected ancestor in the condensed tree.
        let parent_of: BTreeMap<usize, usize> =
            condensed.iter().map(|r| (r.child, r.parent)).collect();
        let index_of: BTreeMap<usize, usize> =
            selected.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let labels: Vec<i64> = (0..rows)
            .map(|p| {
                let mut cur = p;
                while let Some(&par) = parent_of.get(&cur) {
                    if let Some(&label) = index_of.get(&par) {
                        return label as i64;
                    }
                    cur = par;
                }
                NOISE
            })
            .collect();

        // Each selected cluster is a non-empty set of samples, and every label names one.
        ensures!(
            labels
                .iter()
                .all(|&l| l == NOISE || (l as usize) < selected.len()),
            "hdbscan.labels_valid"
        );
        self.labels = Some(labels);
        Ok(())
    }
}

// ---------------------------------------------------------------- settings

impl<F: Float> Hdbscan<F> {
    /// The minimum cluster size the model was configured with.
    pub fn min_cluster_size(&self) -> usize {
        self.min_cluster_size
    }

    /// The `min_samples` the model was configured with, or `None` for its default.
    pub fn min_samples(&self) -> Option<usize> {
        self.min_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ml_proven_core::DenseView;

    #[test]
    fn two_separated_blobs_and_noise() {
        // Two tight groups of 6 points, plus one isolated point.
        let mut data = Vec::new();
        for i in 0..6 {
            data.extend([i as f64 * 0.01, 0.0]);
        }
        for i in 0..6 {
            data.extend([10.0 + i as f64 * 0.01, 0.0]);
        }
        data.extend([50.0, 50.0]);
        let mut m = Hdbscan::<f64>::new(3);
        UnsupervisedEstimator::fit(&mut m, DenseView::new(&data, 13, 2).unwrap()).unwrap();
        let labels = m.labels().unwrap();
        assert_eq!(m.n_clusters(), Some(2));
        assert!(labels[..6].iter().all(|&l| l == labels[0] && l != NOISE));
        assert!(labels[6..12].iter().all(|&l| l == labels[6] && l != NOISE));
        assert_ne!(labels[0], labels[6]);
        assert_eq!(labels[12], NOISE);
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        let data = [0.0, 1.0, 2.0];
        let mut m = Hdbscan::<f64>::new(1);
        assert!(matches!(
            UnsupervisedEstimator::fit(&mut m, DenseView::new(&data, 3, 1).unwrap()),
            Err(Error::InvalidParameter {
                name: "min_cluster_size",
                ..
            })
        ));
        let mut m = Hdbscan::<f64>::new(2).with_min_samples(4);
        assert!(matches!(
            UnsupervisedEstimator::fit(&mut m, DenseView::new(&data, 3, 1).unwrap()),
            Err(Error::InvalidParameter {
                name: "min_samples",
                ..
            })
        ));
    }
}
