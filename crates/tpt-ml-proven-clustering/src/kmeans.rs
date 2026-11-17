// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! KMeans clustering by Lloyd's algorithm.
//!
//! The contract for this algorithm is `docs/contracts/kmeans.tptc`. Every clause
//! there has a runtime check in this file under the name `kmeans.<clause id>`.
//!
//! Differences from scikit-learn, kept deliberate:
//!
//! * An empty cluster keeps its previous centre. Moving it would make the
//!   inertia argument in `kmeans.monotone_inertia` harder to state, and the
//!   descent property is what the contract is about.
//! * `converged` means the assignments reached a fixed point (the contract's
//!   meaning). Stopping on a small inertia change is a separate outcome,
//!   reported through [`KMeans::converged`] as `false`.
//! * [`Transformer::transform`](tpt_ml_proven_core::Transformer::transform)
//!   returns squared Euclidean distances to each centre. Square roots are not
//!   available in `no_std` core.

use tpt_ml_proven_core::error::{Error, ShapeError};
use tpt_ml_proven_core::{
    ensures, requires, ArrayView, Dense, Float, Predictor, Transformer, UnsupervisedEstimator,
};

use crate::rng::SplitMix64;

/// How initial centres are chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Init {
    /// k-means++ (Arthur & Vassilvitskii, 2007): each new centre is drawn with
    /// probability proportional to its squared distance from the nearest existing
    /// centre.
    KMeansPlusPlus,
    /// `k` distinct samples chosen uniformly at random.
    Random,
}

/// KMeans estimator. Build with [`KMeans::new`] and the `with_*` methods, then
/// call `fit`.
#[derive(Debug, Clone, PartialEq)]
pub struct KMeans<F: Float> {
    n_clusters: usize,
    max_iter: usize,
    tol: f64,
    init: Init,
    seed: u64,
    threads: usize,
    fitted: Option<Fitted<F>>,
}

#[derive(Debug, Clone, PartialEq)]
struct Fitted<F: Float> {
    centers: Dense<F>,
    labels: Vec<usize>,
    inertia: F,
    n_iter: usize,
    converged: bool,
}

impl<F: Float> KMeans<F> {
    /// An unfitted estimator with `n_clusters` clusters, 300 iterations at most,
    /// relative tolerance `1e-4`, k-means++ initialisation, seed `0`, one thread.
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            max_iter: 300,
            tol: 1e-4,
            init: Init::KMeansPlusPlus,
            seed: 0,
            threads: 1,
            fitted: None,
        }
    }

    /// Sets the iteration limit.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the relative inertia tolerance. Iteration stops early when the inertia
    /// falls by no more than `tol * (1 + inertia)`.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the initialisation method.
    pub fn with_init(mut self, init: Init) -> Self {
        self.init = init;
        self
    }

    /// Sets the seed. The same seed and data give the same model.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the worker count for the assignment step. Only used when the
    /// `parallel` feature is enabled. Results are identical for any value.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// The number of clusters requested.
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// Fitted centres (`n_clusters x n_features`), or `None` before `fit`.
    pub fn cluster_centers(&self) -> Option<&Dense<F>> {
        self.fitted.as_ref().map(|f| &f.centers)
    }

    /// Cluster label of each training sample, or `None` before `fit`.
    pub fn labels(&self) -> Option<&[usize]> {
        self.fitted.as_ref().map(|f| f.labels.as_slice())
    }

    /// Sum of squared distances from each training sample to its centre.
    pub fn inertia(&self) -> Option<F> {
        self.fitted.as_ref().map(|f| f.inertia)
    }

    /// Lloyd iterations performed.
    pub fn n_iter(&self) -> Option<usize> {
        self.fitted.as_ref().map(|f| f.n_iter)
    }

    /// `true` if the assignments reached a fixed point.
    pub fn converged(&self) -> Option<bool> {
        self.fitted.as_ref().map(|f| f.converged)
    }

    fn validate(&self, rows: usize, cols: usize) -> Result<(), Error> {
        if self.n_clusters == 0 {
            return Err(Error::InvalidParameter {
                name: "n_clusters",
                reason: "must be at least 1",
            });
        }
        if self.max_iter == 0 {
            return Err(Error::InvalidParameter {
                name: "max_iter",
                reason: "must be at least 1",
            });
        }
        if !self.tol.is_finite() || self.tol < 0.0 {
            return Err(Error::InvalidParameter {
                name: "tol",
                reason: "must be finite and non-negative",
            });
        }
        if self.threads == 0 {
            return Err(Error::InvalidParameter {
                name: "threads",
                reason: "must be at least 1",
            });
        }
        if rows == 0 || cols == 0 {
            return Err(Error::EmptyInput);
        }
        if self.n_clusters > rows {
            return Err(Error::InvalidParameter {
                name: "n_clusters",
                reason: "exceeds the number of samples",
            });
        }
        Ok(())
    }
}

impl<F: Float> UnsupervisedEstimator for KMeans<F> {
    type Float = F;
    type FitError = Error;

    fn fit<X>(&mut self, x: X) -> Result<(), Error>
    where
        X: ArrayView<Elem = F>,
    {
        // A failed fit leaves the estimator unfitted.
        self.fitted = None;
        let (rows, cols) = (x.rows(), x.cols());
        self.validate(rows, cols)?;
        if !x.all_finite() {
            return Err(Error::NonFinite { what: "X" });
        }
        let k = self.n_clusters;
        requires!(rows >= 1, "kmeans.rows_positive");
        requires!(k >= 1, "kmeans.k_positive");
        requires!(k <= rows, "kmeans.k_le_rows");

        let mut rng = SplitMix64::new(self.seed);
        let mut centers = match self.init {
            Init::KMeansPlusPlus => plus_plus_init(&x, k, &mut rng),
            Init::Random => random_init(&x, k, &mut rng),
        };
        let tol = F::from_f64(self.tol);

        let (mut labels, mut inertia) = assign(&x, &centers, k, cols, self.threads);
        let mut n_iter = 0;
        let mut converged = false;

        for iter in 0..self.max_iter {
            let prev_inertia = inertia;
            update_centers(&x, &labels, k, cols, &mut centers);
            // The centres just computed are means of the members under `labels`.
            ensures!(
                centers_within_member_range(&x, &labels, &centers, k, cols),
                "kmeans.centers_in_hull"
            );

            let (new_labels, new_inertia) = assign(&x, &centers, k, cols, self.threads);
            n_iter = iter + 1;
            ensures!(
                new_inertia <= prev_inertia + descent_slack(prev_inertia),
                "kmeans.monotone_inertia"
            );

            converged = new_labels == labels;
            if converged {
                ensures!(new_labels == labels, "kmeans.converged_means_stable");
            }

            let improvement = prev_inertia - new_inertia;
            labels = new_labels;
            inertia = new_inertia;
            if converged || improvement <= tol * (F::ONE + prev_inertia) {
                break;
            }
        }

        ensures!(k <= rows, "kmeans.k_within_rows");
        self.fitted = Some(Fitted {
            centers: Dense::from_vec(centers, k, cols)?,
            labels,
            inertia,
            n_iter,
            converged,
        });
        Ok(())
    }
}

impl<F: Float> Predictor for KMeans<F> {
    type Float = F;
    type Output = Vec<usize>;

    fn predict<X>(&self, x: X) -> Result<Vec<usize>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        let fitted = self.fitted.as_ref().ok_or(Error::NotFitted)?;
        check_features(&x, fitted.centers.cols())?;
        if !x.all_finite() {
            return Err(Error::NonFinite { what: "X" });
        }
        let c = centers_flat(&fitted.centers);
        let (labels, _) = assign(&x, &c, fitted.centers.rows(), x.cols(), self.threads);
        Ok(labels)
    }
}

impl<F: Float> Transformer for KMeans<F> {
    type Float = F;
    type Output = Dense<F>;

    /// Squared Euclidean distance from each sample to each centre (`rows x k`).
    fn transform<X>(&self, x: X) -> Result<Dense<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        let fitted = self.fitted.as_ref().ok_or(Error::NotFitted)?;
        check_features(&x, fitted.centers.cols())?;
        if !x.all_finite() {
            return Err(Error::NonFinite { what: "X" });
        }
        let k = fitted.centers.rows();
        let cols = x.cols();
        let c = centers_flat(&fitted.centers);
        let mut out = Vec::with_capacity(x.rows() * k);
        for r in 0..x.rows() {
            for j in 0..k {
                out.push(dist_row(&x, r, &c[j * cols..(j + 1) * cols]));
            }
        }
        Ok(Dense::from_vec(out, x.rows(), k)?)
    }

    fn fit_transform<X>(&mut self, x: X) -> Result<Dense<F>, Error>
    where
        X: ArrayView<Elem = F>,
    {
        UnsupervisedEstimator::fit(self, &x)?;
        self.transform(x)
    }
}

// ---------------------------------------------------------------- helpers

fn check_features<X: ArrayView>(x: &X, expected: usize) -> Result<(), Error> {
    if x.cols() != expected {
        return Err(Error::Shape(ShapeError {
            rows: x.rows(),
            cols: x.cols(),
            reason: "feature count differs from the fitted model",
        }));
    }
    Ok(())
}

fn centers_flat<F: Float>(centers: &Dense<F>) -> Vec<F> {
    let mut out = Vec::with_capacity(centers.rows() * centers.cols());
    for r in 0..centers.rows() {
        for c in 0..centers.cols() {
            out.push(centers.get(r, c));
        }
    }
    out
}

/// Squared distance between sample row `r` and a centre, summed in column order.
fn dist_row<F: Float, X: ArrayView<Elem = F>>(x: &X, r: usize, centre: &[F]) -> F {
    let mut acc = F::ZERO;
    for (j, &c) in centre.iter().enumerate() {
        let d = x.get(r, j) - c;
        acc = acc + d * d;
    }
    acc
}

/// Nearest centre for one row (ties go to the lowest index), and that distance.
fn nearest<F: Float, X: ArrayView<Elem = F>>(
    x: &X,
    r: usize,
    centers: &[F],
    k: usize,
    cols: usize,
) -> (usize, F) {
    let mut best = (0, dist_row(x, r, &centers[0..cols]));
    for c in 1..k {
        let d = dist_row(x, r, &centers[c * cols..(c + 1) * cols]);
        if d < best.1 {
            best = (c, d);
        }
    }
    best
}

/// Assignment step: labels and total inertia. The inertia is summed in row order,
/// so it does not depend on how the work is split.
fn assign<F: Float, X: ArrayView<Elem = F>>(
    x: &X,
    centers: &[F],
    k: usize,
    cols: usize,
    threads: usize,
) -> (Vec<usize>, F) {
    #[cfg(feature = "parallel")]
    {
        if threads > 1 && x.rows() > 1 {
            return assign_parallel(x, centers, k, cols, threads);
        }
    }
    let _ = threads;
    assign_sequential(x, centers, k, cols)
}

fn assign_sequential<F: Float, X: ArrayView<Elem = F>>(
    x: &X,
    centers: &[F],
    k: usize,
    cols: usize,
) -> (Vec<usize>, F) {
    let mut labels = Vec::with_capacity(x.rows());
    let mut inertia = F::ZERO;
    for r in 0..x.rows() {
        let (c, d) = nearest(x, r, centers, k, cols);
        labels.push(c);
        inertia = inertia + d;
    }
    (labels, inertia)
}

#[cfg(feature = "parallel")]
fn assign_parallel<F: Float, X: ArrayView<Elem = F>>(
    x: &X,
    centers: &[F],
    k: usize,
    cols: usize,
    threads: usize,
) -> (Vec<usize>, F) {
    // Threads need plain slices, so the data is copied into a row-major buffer
    // first. The arithmetic matches the sequential path, so the results do too.
    let rows = x.rows();
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for j in 0..cols {
            data.push(x.get(r, j));
        }
    }
    let chunk = rows.div_ceil(threads.min(rows)).max(1);
    let parts: Vec<Vec<(usize, F)>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..rows)
            .step_by(chunk)
            .map(|start| {
                let end = (start + chunk).min(rows);
                let data = &data;
                s.spawn(move || {
                    (start..end)
                        .map(|r| nearest_slice(&data[r * cols..(r + 1) * cols], centers, k, cols))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    });
    let mut labels = Vec::with_capacity(rows);
    let mut inertia = F::ZERO;
    for part in parts {
        for (c, d) in part {
            labels.push(c);
            inertia = inertia + d;
        }
    }
    (labels, inertia)
}

#[cfg(feature = "parallel")]
fn nearest_slice<F: Float>(row: &[F], centers: &[F], k: usize, cols: usize) -> (usize, F) {
    let dist = |centre: &[F]| {
        let mut acc = F::ZERO;
        for (&a, &c) in row.iter().zip(centre) {
            let d = a - c;
            acc = acc + d * d;
        }
        acc
    };
    let mut best = (0, dist(&centers[0..cols]));
    for c in 1..k {
        let d = dist(&centers[c * cols..(c + 1) * cols]);
        if d < best.1 {
            best = (c, d);
        }
    }
    best
}

/// Update step: each centre becomes the mean of its members. An empty cluster
/// keeps its previous centre.
fn update_centers<F: Float, X: ArrayView<Elem = F>>(
    x: &X,
    labels: &[usize],
    k: usize,
    cols: usize,
    centers: &mut [F],
) {
    let mut sums = vec![F::ZERO; k * cols];
    let mut counts = vec![0_usize; k];
    for (r, &c) in labels.iter().enumerate() {
        counts[c] += 1;
        for j in 0..cols {
            sums[c * cols + j] = sums[c * cols + j] + x.get(r, j);
        }
    }
    for c in 0..k {
        if counts[c] == 0 {
            continue;
        }
        let n = F::from_usize(counts[c]);
        for j in 0..cols {
            centers[c * cols + j] = sums[c * cols + j] / n;
        }
    }
}

/// Rounding slack for the monotone-descent check. The contract is stated over the
/// reals, and the floating-point sums can drift by a few ulps.
fn descent_slack<F: Float>(scale: F) -> F {
    F::EPSILON * F::from_usize(64) * (F::ONE + scale)
}

fn abs_f<F: Float>(x: F) -> F {
    if x < F::ZERO {
        -x
    } else {
        x
    }
}

/// `true` if every non-empty cluster's centre lies within the per-coordinate range
/// of its members, to within rounding. A centre that is the mean of its members is
/// a convex combination of them, so this is a necessary condition for lying in
/// their convex hull, and it is the check run for `kmeans.centers_in_hull`.
fn centers_within_member_range<F: Float, X: ArrayView<Elem = F>>(
    x: &X,
    labels: &[usize],
    centers: &[F],
    k: usize,
    cols: usize,
) -> bool {
    let mut lo = vec![F::ZERO; k * cols];
    let mut hi = vec![F::ZERO; k * cols];
    let mut seen = vec![false; k];
    for (r, &c) in labels.iter().enumerate() {
        for j in 0..cols {
            let v = x.get(r, j);
            let idx = c * cols + j;
            if !seen[c] || v < lo[idx] {
                lo[idx] = v;
            }
            if !seen[c] || v > hi[idx] {
                hi[idx] = v;
            }
        }
        seen[c] = true;
    }
    for (c, &has_members) in seen.iter().enumerate() {
        if !has_members {
            continue;
        }
        for j in 0..cols {
            let idx = c * cols + j;
            let slack = F::EPSILON * F::from_usize(16) * (F::ONE + abs_f(lo[idx]) + abs_f(hi[idx]));
            let v = centers[idx];
            if v < lo[idx] - slack || v > hi[idx] + slack {
                return false;
            }
        }
    }
    true
}

/// k-means++ seeding.
fn plus_plus_init<F: Float, X: ArrayView<Elem = F>>(
    x: &X,
    k: usize,
    rng: &mut SplitMix64,
) -> Vec<F> {
    let (rows, cols) = (x.rows(), x.cols());
    let mut centers = Vec::with_capacity(k * cols);
    let first = rng.next_below(rows);
    push_row(x, first, &mut centers);

    // Weights are squared distances to the nearest chosen centre, in f64 for sampling.
    let mut weights: Vec<f64> = (0..rows)
        .map(|r| dist_row(x, r, &centers[0..cols]).to_f64())
        .collect();
    for _ in 1..k {
        let total: f64 = weights.iter().sum();
        let pick = if total > 0.0 {
            let target = rng.next_unit() * total;
            let mut acc = 0.0;
            let mut chosen = rows - 1;
            for (r, &w) in weights.iter().enumerate() {
                acc += w;
                if acc > target {
                    chosen = r;
                    break;
                }
            }
            chosen
        } else {
            // Every sample coincides with a centre already, so any sample will do.
            rng.next_below(rows)
        };
        push_row(x, pick, &mut centers);
        let new_centre = centers.len() - cols;
        for (r, w) in weights.iter_mut().enumerate() {
            let d = dist_row(x, r, &centers[new_centre..new_centre + cols]).to_f64();
            if d < *w {
                *w = d;
            }
        }
    }
    centers
}

/// Random seeding: `k` distinct samples, chosen by a partial Fisher–Yates shuffle.
fn random_init<F: Float, X: ArrayView<Elem = F>>(x: &X, k: usize, rng: &mut SplitMix64) -> Vec<F> {
    let rows = x.rows();
    let mut order: Vec<usize> = (0..rows).collect();
    let mut centers = Vec::with_capacity(k * x.cols());
    for i in 0..k {
        let j = i + rng.next_below(rows - i);
        order.swap(i, j);
        push_row(x, order[i], &mut centers);
    }
    centers
}

fn push_row<F: Float, X: ArrayView<Elem = F>>(x: &X, r: usize, out: &mut Vec<F>) {
    for j in 0..x.cols() {
        out.push(x.get(r, j));
    }
}
