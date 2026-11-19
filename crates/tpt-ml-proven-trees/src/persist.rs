// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Model files for the tree models, in the format described by `tpt_ml_proven_core::persist`.
//!
//! A tree is stored as its node list. Loading checks every child index, feature index and
//! leaf count, so a file that passes the checksum but describes a malformed tree is refused.

use tpt_ml_proven_core::error::Error;
use tpt_ml_proven_core::persist::{dtype, kind, Reader, Writer};

use crate::classifier::{check_params, Criterion, DecisionTreeClassifier, Fitted};
use crate::forest::{ForestFit, RandomForestClassifier, RandomForestRegressor};
use crate::gbm::{Boosted, DecisionTreeRegressor, GradientBoostingRegressor};
use crate::tree::{Node, Tree};

fn invalid(reason: &'static str) -> Error {
    Error::Format { reason }
}

fn write_tree(w: &mut Writer, t: &Tree) {
    w.usize(t.n_features);
    w.usize(t.depth);
    w.usize(t.nodes.len());
    for node in &t.nodes {
        match *node {
            Node::Leaf { value, count } => {
                w.usize(0);
                w.f64(value);
                w.usize(count);
            }
            Node::Split {
                feature,
                threshold,
                left,
                right,
                count,
            } => {
                w.usize(1);
                w.usize(feature);
                w.f64(threshold);
                w.usize(left);
                w.usize(right);
                w.usize(count);
            }
        }
    }
}

fn read_tree(r: &mut Reader<'_>) -> Result<Tree, Error> {
    let n_features = r.usize()?;
    let depth = r.usize()?;
    let n_nodes = r.usize()?;
    if n_nodes == 0 {
        return Err(invalid("tree has no nodes"));
    }
    let mut nodes = Vec::new();
    for _ in 0..n_nodes {
        let node = match r.usize()? {
            0 => Node::Leaf {
                value: r.f64()?,
                count: r.usize()?,
            },
            1 => {
                let feature = r.usize()?;
                let threshold = r.f64()?;
                let left = r.usize()?;
                let right = r.usize()?;
                let count = r.usize()?;
                if feature >= n_features {
                    return Err(invalid("tree splits on a feature that does not exist"));
                }
                if left >= n_nodes || right >= n_nodes {
                    return Err(invalid("tree child index is out of range"));
                }
                Node::Split {
                    feature,
                    threshold,
                    left,
                    right,
                    count,
                }
            }
            _ => return Err(invalid("unknown tree node type")),
        };
        nodes.push(node);
    }
    let tree = Tree {
        nodes,
        n_features,
        depth,
        importances: Vec::new(),
    };
    if tree.leaf_count_total() == 0 {
        return Err(invalid("tree has no training rows in its leaves"));
    }
    Ok(tree)
}

/// The kind tag of a model file, read from its header. `None` if the bytes are too short.
fn file_kind(bytes: &[u8]) -> Result<u32, Error> {
    bytes
        .get(12..16)
        .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
        .ok_or_else(|| invalid("not a tree model file"))
}

/// The per-feature importances stored after a tree. Empty means none were stored.
fn read_importances(r: &mut Reader<'_>, n_features: usize) -> Result<Vec<f64>, Error> {
    let values = r.f64s()?;
    if !values.is_empty() && values.len() != n_features {
        return Err(invalid("tree importances do not match the feature count"));
    }
    if values.iter().any(|g| !g.is_finite() || *g < 0.0) {
        return Err(invalid("tree importances are not finite and non-negative"));
    }
    Ok(values)
}

impl DecisionTreeRegressor {
    /// Serialises the model to a model file. A model that uses none of the newer settings is
    /// written in the older kind, so loading and saving an older file leaves its bytes unchanged.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let uses_new = self.min_samples_split() != 2
            || self.max_leaf_nodes().is_some()
            || self.tree().is_some_and(|t| !t.importances.is_empty());
        if !uses_new {
            return self.to_bytes_v1();
        }
        let mut w = Writer::with_dtype(kind::DECISION_TREE_V2, dtype::F64);
        w.usize(self.max_depth());
        w.usize(self.min_samples_leaf());
        w.usize(self.min_samples_split());
        w.usize(self.max_leaf_nodes().unwrap_or(0));
        match self.tree() {
            None => w.bool(false),
            Some(t) => {
                w.bool(true);
                write_tree(&mut w, t);
                w.f64s(&t.importances);
            }
        }
        Ok(w.finish())
    }

    /// Writes the older kind 8 layout: depth, leaf size and the tree.
    fn to_bytes_v1(&self) -> Result<Vec<u8>, Error> {
        let mut w = Writer::with_dtype(kind::DECISION_TREE, dtype::F64);
        w.usize(self.max_depth());
        w.usize(self.min_samples_leaf());
        match self.tree() {
            None => w.bool(false),
            Some(t) => {
                w.bool(true);
                write_tree(&mut w, t);
            }
        }
        Ok(w.finish())
    }

    /// Loads a model written by [`DecisionTreeRegressor::to_bytes`]. Files written before the
    /// leaf limit, the split size and the importances were stored load with the defaults.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if file_kind(bytes)? == kind::DECISION_TREE {
            return Self::from_bytes_v1(bytes);
        }
        let mut r = Reader::new(bytes, kind::DECISION_TREE_V2)?;
        let max_depth = r.usize()?;
        let min_samples_leaf = r.usize()?;
        let min_samples_split = r.usize()?;
        let max_leaf_nodes = match r.usize()? {
            0 => None,
            k => Some(k),
        };
        let tree = if r.bool()? {
            let mut tree = read_tree(&mut r)?;
            tree.importances = read_importances(&mut r, tree.n_features)?;
            Some(tree)
        } else {
            None
        };
        r.finish()?;
        check_params(
            max_depth,
            min_samples_leaf,
            min_samples_split,
            max_leaf_nodes,
        )?;
        Ok(DecisionTreeRegressor::from_parts(
            max_depth,
            min_samples_leaf,
            min_samples_split,
            max_leaf_nodes,
            tree,
        ))
    }

    /// Reads a kind 8 file: depth, leaf size and the tree, with no importances.
    fn from_bytes_v1(bytes: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(bytes, kind::DECISION_TREE)?;
        let max_depth = r.usize()?;
        let min_samples_leaf = r.usize()?;
        let tree = if r.bool()? {
            Some(read_tree(&mut r)?)
        } else {
            None
        };
        r.finish()?;
        if max_depth == 0 || min_samples_leaf == 0 {
            return Err(invalid("tree parameters must be at least 1"));
        }
        Ok(DecisionTreeRegressor::from_parts(
            max_depth,
            min_samples_leaf,
            2,
            None,
            tree,
        ))
    }
}

impl GradientBoostingRegressor {
    /// Serialises the model to a model file. Writes the sampling settings too, under kind 22.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut w = Writer::with_dtype(kind::GRADIENT_BOOSTING_SAMPLED, dtype::F64);
        w.usize(self.n_estimators());
        w.f64(self.learning_rate());
        w.usize(self.max_depth());
        w.usize(self.min_samples_leaf());
        w.f64(self.subsample());
        match self.max_features() {
            None => w.bool(false),
            Some(m) => {
                w.bool(true);
                w.usize(m);
            }
        }
        w.u64(self.seed());
        match self.boosted() {
            None => w.bool(false),
            Some(b) => {
                w.bool(true);
                w.f64(b.init);
                w.usize(b.n_features);
                w.f64s(&b.train_loss);
                w.usize(b.trees.len());
                for t in &b.trees {
                    write_tree(&mut w, t);
                }
            }
        }
        Ok(w.finish())
    }

    /// Loads a model written by [`GradientBoostingRegressor::to_bytes`]. Files written before
    /// the sampling settings existed (kind 9) load with the defaults: every row, every feature,
    /// seed 0.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let sampled = matches!(
            bytes.get(12..16).map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes"))),
            Some(k) if k == kind::GRADIENT_BOOSTING_SAMPLED
        );
        let tag = if sampled {
            kind::GRADIENT_BOOSTING_SAMPLED
        } else {
            kind::GRADIENT_BOOSTING
        };
        let mut r = Reader::new(bytes, tag)?;
        let n_estimators = r.usize()?;
        let learning_rate = r.f64()?;
        let max_depth = r.usize()?;
        let min_samples_leaf = r.usize()?;
        let (subsample, max_features, seed) = if sampled {
            let subsample = r.f64()?;
            let max_features = if r.bool()? { Some(r.usize()?) } else { None };
            (subsample, max_features, r.u64()?)
        } else {
            (1.0, None, 0)
        };
        let boosted = if r.bool()? {
            let init = r.f64()?;
            let n_features = r.usize()?;
            let train_loss = r.f64s()?;
            let n_trees = r.usize()?;
            let mut trees = Vec::new();
            for _ in 0..n_trees {
                trees.push(read_tree(&mut r)?);
            }
            if trees.iter().any(|t| t.n_features != n_features) {
                return Err(invalid(
                    "gradient boosting trees disagree on the feature count",
                ));
            }
            Some(Boosted {
                init,
                trees,
                n_features,
                train_loss,
                learning_rate,
            })
        } else {
            None
        };
        r.finish()?;
        if !(learning_rate.is_finite() && learning_rate > 0.0 && learning_rate <= 1.0) {
            return Err(invalid("gradient boosting learning rate is outside (0, 1]"));
        }
        if n_estimators == 0 || max_depth == 0 || min_samples_leaf == 0 {
            return Err(invalid("gradient boosting parameters must be at least 1"));
        }
        if !(subsample > 0.0 && subsample <= 1.0) {
            return Err(invalid("gradient boosting subsample is outside (0, 1]"));
        }
        if max_features == Some(0) {
            return Err(invalid("gradient boosting max_features must be at least 1"));
        }
        Ok(GradientBoostingRegressor::from_parts(
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            (subsample, max_features, seed),
            boosted,
        ))
    }
}

// ---------------------------------------------------------------- classifier

impl DecisionTreeClassifier {
    /// Serialises the classifier (settings and fitted tree) to a model file. A classifier that
    /// uses none of the newer settings is written in the older kind, so loading and saving an
    /// older file leaves its bytes unchanged.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let uses_new = self.min_samples_split() != 2
            || self.max_leaf_nodes().is_some()
            || self
                .fitted
                .as_ref()
                .is_some_and(|f| !f.tree.importances.is_empty());
        if !uses_new {
            return self.to_bytes_v1();
        }
        let mut w = Writer::for_float::<f64>(kind::DECISION_TREE_CLASSIFIER_V2);
        w.usize(self.max_depth);
        w.usize(self.min_samples_leaf);
        w.usize(self.min_samples_split);
        w.usize(self.max_leaf_nodes.unwrap_or(0));
        w.bool(self.criterion == Criterion::Entropy);
        match &self.fitted {
            None => w.bool(false),
            Some(f) => {
                w.bool(true);
                w.f64s(&f.classes);
                write_tree(&mut w, &f.tree);
                w.f64s(&f.tree.importances);
            }
        }
        Ok(w.finish())
    }

    /// Loads a classifier written by [`DecisionTreeClassifier::to_bytes`]. Checks the settings,
    /// that the class labels are sorted and distinct, and that every leaf predicts one of them.
    /// Files written before the leaf limit, the split size and the importances were stored load
    /// with the defaults.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if file_kind(bytes)? == kind::DECISION_TREE_CLASSIFIER {
            return Self::from_bytes_v1(bytes);
        }
        let mut r = Reader::new(bytes, kind::DECISION_TREE_CLASSIFIER_V2)?;
        let max_depth = r.usize()?;
        let min_samples_leaf = r.usize()?;
        let min_samples_split = r.usize()?;
        let max_leaf_nodes = match r.usize()? {
            0 => None,
            k => Some(k),
        };
        let criterion = if r.bool()? {
            Criterion::Entropy
        } else {
            Criterion::Gini
        };
        let fitted = if r.bool()? {
            let classes = r.f64s()?;
            let mut tree = read_tree(&mut r)?;
            tree.importances = read_importances(&mut r, tree.n_features)?;
            Some(Fitted { tree, classes })
        } else {
            None
        };
        r.finish()?;
        classifier_from_parts(ClassifierParts {
            max_depth,
            min_samples_leaf,
            min_samples_split,
            max_leaf_nodes,
            criterion,
            fitted,
        })
    }

    /// Writes the older kind 19 layout: depth, leaf size, criterion, classes and tree.
    fn to_bytes_v1(&self) -> Result<Vec<u8>, Error> {
        let mut w = Writer::for_float::<f64>(kind::DECISION_TREE_CLASSIFIER);
        w.usize(self.max_depth);
        w.usize(self.min_samples_leaf);
        w.bool(self.criterion == Criterion::Entropy);
        match &self.fitted {
            None => w.bool(false),
            Some(f) => {
                w.bool(true);
                w.f64s(&f.classes);
                write_tree(&mut w, &f.tree);
            }
        }
        Ok(w.finish())
    }

    /// Reads a kind 19 file: the depth, leaf size, criterion and tree, with no importances.
    fn from_bytes_v1(bytes: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(bytes, kind::DECISION_TREE_CLASSIFIER)?;
        let max_depth = r.usize()?;
        let min_samples_leaf = r.usize()?;
        let criterion = if r.bool()? {
            Criterion::Entropy
        } else {
            Criterion::Gini
        };
        let fitted = if r.bool()? {
            let classes = r.f64s()?;
            let tree = read_tree(&mut r)?;
            Some(Fitted { tree, classes })
        } else {
            None
        };
        r.finish()?;
        classifier_from_parts(ClassifierParts {
            max_depth,
            min_samples_leaf,
            min_samples_split: 2,
            max_leaf_nodes: None,
            criterion,
            fitted,
        })
    }
}

/// The stored parts of a classifier, before they are checked.
struct ClassifierParts {
    max_depth: usize,
    min_samples_leaf: usize,
    min_samples_split: usize,
    max_leaf_nodes: Option<usize>,
    criterion: Criterion,
    fitted: Option<Fitted>,
}

/// Checks the stored parts of a classifier and builds it.
fn classifier_from_parts(p: ClassifierParts) -> Result<DecisionTreeClassifier, Error> {
    check_params(
        p.max_depth,
        p.min_samples_leaf,
        p.min_samples_split,
        p.max_leaf_nodes,
    )?;
    if let Some(f) = &p.fitted {
        if f.classes.is_empty() || f.classes.windows(2).any(|w| w[0] >= w[1]) {
            return Err(invalid("classifier classes are not sorted and distinct"));
        }
        let known = f.tree.nodes.iter().all(|n| match n {
            Node::Leaf { value, .. } => f.classes.contains(value),
            Node::Split { .. } => true,
        });
        if !known {
            return Err(invalid(
                "a classifier leaf predicts a label that was never seen",
            ));
        }
    }
    Ok(DecisionTreeClassifier {
        max_depth: p.max_depth,
        min_samples_leaf: p.min_samples_leaf,
        min_samples_split: p.min_samples_split,
        max_leaf_nodes: p.max_leaf_nodes,
        criterion: p.criterion,
        fitted: p.fitted,
    })
}

// ---------------------------------------------------------------- forests

/// Writes the hyper-parameters shared by both forests.
fn write_forest_settings(
    w: &mut Writer,
    n_estimators: usize,
    max_depth: usize,
    min_samples_leaf: usize,
    max_features: Option<usize>,
    seed: u64,
) {
    w.usize(n_estimators);
    w.usize(max_depth);
    w.usize(min_samples_leaf);
    match max_features {
        None => w.bool(false),
        Some(m) => {
            w.bool(true);
            w.usize(m);
        }
    }
    w.u64(seed);
}

/// Reads and checks the hyper-parameters written by [`write_forest_settings`].
fn read_forest_settings(
    r: &mut Reader<'_>,
) -> Result<(usize, usize, usize, Option<usize>, u64), Error> {
    let n_estimators = r.usize()?;
    let max_depth = r.usize()?;
    let min_samples_leaf = r.usize()?;
    let max_features = if r.bool()? { Some(r.usize()?) } else { None };
    let seed = r.u64()?;
    if n_estimators == 0 || max_depth == 0 || min_samples_leaf == 0 {
        return Err(invalid("forest settings must be at least 1"));
    }
    if max_features == Some(0) {
        return Err(invalid("forest max_features must be at least 1"));
    }
    Ok((
        n_estimators,
        max_depth,
        min_samples_leaf,
        max_features,
        seed,
    ))
}

/// Reads the trees of a fitted forest, checking their count and their feature width.
fn read_forest_trees(
    r: &mut Reader<'_>,
    n_estimators: usize,
    n_features: usize,
) -> Result<Vec<Tree>, Error> {
    let count = r.usize()?;
    if count != n_estimators {
        return Err(invalid("forest tree count differs from n_estimators"));
    }
    let mut trees = Vec::new();
    for _ in 0..count {
        let t = read_tree(r)?;
        if t.n_features != n_features {
            return Err(invalid("forest tree width differs from the forest's"));
        }
        trees.push(t);
    }
    Ok(trees)
}

impl RandomForestRegressor {
    /// Serialises the forest (settings and fitted trees) to a model file.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut w = Writer::for_float::<f64>(kind::RANDOM_FOREST_REGRESSOR);
        write_forest_settings(
            &mut w,
            self.n_estimators,
            self.max_depth,
            self.min_samples_leaf,
            self.max_features,
            self.seed,
        );
        match &self.fitted {
            None => w.bool(false),
            Some((trees, n_features)) => {
                w.bool(true);
                w.usize(*n_features);
                w.usize(trees.len());
                for t in trees {
                    write_tree(&mut w, t);
                }
            }
        }
        Ok(w.finish())
    }

    /// Loads a forest written by [`RandomForestRegressor::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(bytes, kind::RANDOM_FOREST_REGRESSOR)?;
        let (n_estimators, max_depth, min_samples_leaf, max_features, seed) =
            read_forest_settings(&mut r)?;
        let fitted = if r.bool()? {
            let n_features = r.usize()?;
            let trees = read_forest_trees(&mut r, n_estimators, n_features)?;
            Some((trees, n_features))
        } else {
            None
        };
        r.finish()?;
        Ok(Self {
            n_estimators,
            max_depth,
            min_samples_leaf,
            max_features,
            seed,
            fitted,
        })
    }
}

impl RandomForestClassifier {
    /// Serialises the forest (settings, classes and fitted trees) to a model file.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut w = Writer::for_float::<f64>(kind::RANDOM_FOREST_CLASSIFIER);
        write_forest_settings(
            &mut w,
            self.n_estimators,
            self.max_depth,
            self.min_samples_leaf,
            self.max_features,
            self.seed,
        );
        w.bool(self.criterion == Criterion::Entropy);
        match &self.fitted {
            None => w.bool(false),
            Some(f) => {
                w.bool(true);
                w.usize(f.n_features);
                w.f64s(&f.classes);
                w.usize(f.trees.len());
                for t in &f.trees {
                    write_tree(&mut w, t);
                }
            }
        }
        Ok(w.finish())
    }

    /// Loads a forest written by [`RandomForestClassifier::to_bytes`]. Checks the classes are
    /// sorted and distinct, and that every leaf predicts one of them.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(bytes, kind::RANDOM_FOREST_CLASSIFIER)?;
        let (n_estimators, max_depth, min_samples_leaf, max_features, seed) =
            read_forest_settings(&mut r)?;
        let criterion = if r.bool()? {
            Criterion::Entropy
        } else {
            Criterion::Gini
        };
        let fitted = if r.bool()? {
            let n_features = r.usize()?;
            let classes = r.f64s()?;
            let trees = read_forest_trees(&mut r, n_estimators, n_features)?;
            Some(ForestFit {
                trees,
                classes,
                n_features,
            })
        } else {
            None
        };
        r.finish()?;
        if let Some(f) = &fitted {
            if f.classes.is_empty() || f.classes.windows(2).any(|w| w[0] >= w[1]) {
                return Err(invalid("forest classes are not sorted and distinct"));
            }
            let known = f.trees.iter().all(|t| {
                t.nodes.iter().all(|n| match n {
                    Node::Leaf { value, .. } => f.classes.contains(value),
                    Node::Split { .. } => true,
                })
            });
            if !known {
                return Err(invalid(
                    "a forest leaf predicts a label that was never seen",
                ));
            }
        }
        Ok(Self {
            n_estimators,
            max_depth,
            min_samples_leaf,
            criterion,
            max_features,
            seed,
            fitted,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ml_proven_core::{DenseView, Predictor, SupervisedEstimator};

    fn data() -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..40).map(|i| (i as f64) * 0.25).collect();
        let y: Vec<f64> = (0..20)
            .map(|r| if x[r * 2] < 2.5 { 1.0 } else { -2.0 })
            .collect();
        (x, y)
    }

    #[test]
    fn tree_round_trips_and_predicts_the_same() {
        let (x, y) = data();
        let mut m = DecisionTreeRegressor::new(3);
        SupervisedEstimator::fit(&mut m, DenseView::new(&x, 20, 2).unwrap(), &y[..]).unwrap();
        let loaded = DecisionTreeRegressor::from_bytes(&m.to_bytes().unwrap()).unwrap();
        assert_eq!(loaded, m);
        let a = Predictor::predict(&m, DenseView::new(&x, 20, 2).unwrap()).unwrap();
        let b = Predictor::predict(&loaded, DenseView::new(&x, 20, 2).unwrap()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn boosted_model_round_trips() {
        let (x, y) = data();
        let mut m = GradientBoostingRegressor::new(5, 2).with_learning_rate(0.3);
        SupervisedEstimator::fit(&mut m, DenseView::new(&x, 20, 2).unwrap(), &y[..]).unwrap();
        let loaded = GradientBoostingRegressor::from_bytes(&m.to_bytes().unwrap()).unwrap();
        assert_eq!(loaded, m);
    }

    #[test]
    fn a_tree_file_is_not_a_boosted_model() {
        let bytes = DecisionTreeRegressor::new(2).to_bytes().unwrap();
        assert!(GradientBoostingRegressor::from_bytes(&bytes).is_err());
    }
}
