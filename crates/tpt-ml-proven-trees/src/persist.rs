// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Model files for the tree models, in the format described by `tpt_ml_proven_core::persist`.
//!
//! A tree is stored as its node list. Loading checks every child index, feature index and
//! leaf count, so a file that passes the checksum but describes a malformed tree is refused.

use tpt_ml_proven_core::error::Error;
use tpt_ml_proven_core::persist::{dtype, kind, Reader, Writer};

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
    };
    if tree.leaf_count_total() == 0 {
        return Err(invalid("tree has no training rows in its leaves"));
    }
    Ok(tree)
}

impl DecisionTreeRegressor {
    /// Serialises the model to a model file.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
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

    /// Loads a model written by [`DecisionTreeRegressor::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
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
            tree,
        ))
    }
}

impl GradientBoostingRegressor {
    /// Serialises the model to a model file.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut w = Writer::with_dtype(kind::GRADIENT_BOOSTING, dtype::F64);
        w.usize(self.n_estimators());
        w.f64(self.learning_rate());
        w.usize(self.max_depth());
        w.usize(self.min_samples_leaf());
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

    /// Loads a model written by [`GradientBoostingRegressor::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(bytes, kind::GRADIENT_BOOSTING)?;
        let n_estimators = r.usize()?;
        let learning_rate = r.f64()?;
        let max_depth = r.usize()?;
        let min_samples_leaf = r.usize()?;
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
        Ok(GradientBoostingRegressor::from_parts(
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            boosted,
        ))
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
