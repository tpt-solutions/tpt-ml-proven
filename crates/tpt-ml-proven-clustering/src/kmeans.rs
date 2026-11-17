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
//! * [`Transformer::transform`]
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
    initial_centers: Option<Vec<F>>,
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
            initial_centers: None,
            fitted: None,
        }
    }

    /// Uses these starting centres instead of `init`. `centers` is row-major,
    /// `n_clusters x n_features`. Its length is checked at `fit`. Intended for
    /// reproducing another implementation's run, and for tests.
    pub fn with_initial_centers(mut self, centers: Vec<F>) -> Self {
        self.initial_centers = Some(centers);
        self
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

        let mut centers = match &self.initial_centers {
            Some(given) => {
                if given.len() != k * cols {
                    return Err(Error::InvalidParameter {
                        name: "initial_centers",
                        reason: "length must be n_clusters * n_features",
                    });
                }
                given.clone()
            }
            None => {
                let mut rng = SplitMix64::new(self.seed);
                match self.init {
                    Init::KMeansPlusPlus => plus_plus_init(&x, k, &mut rng),
                    Init::Random => random_init(&x, k, &mut rng),
                }
            }
        };
        let tol = F::from_f64(self.tol);

        let dense = dense_copy_if_parallel(&x, self.threads);
        let dense = dense.as_deref();
        let (mut labels, mut inertia) = assign(&x, dense, &centers, k, cols, self.threads);
        let mut n_iter = 0;
        let mut converged = false;

        for iter in 0..self.max_iter {
            let prev_inertia = inertia;
            let extent = update_centers(&x, dense, &labels, k, cols, &mut centers, self.threads);
            // The centres just computed are means of the members under `labels`.
            ensures!(
                within_extent(&centers, &extent, k, cols),
                "kmeans.centers_in_hull"
            );

            let (new_labels, new_inertia) = assign(&x, dense, &centers, k, cols, self.threads);
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
        let (labels, _) = assign(&x, None, &c, fitted.centers.rows(), x.cols(), self.threads);
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
        for_each_row(&x, |_, row| {
            for j in 0..k {
                out.push(sq_dist(row, &c[j * cols..(j + 1) * cols]));
            }
        });
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

/// Squared distance between two rows of equal length.
///
/// Eight independent accumulators let the compiler vectorise the loop. The order
/// of summation is fixed, so the result is reproducible. Every path uses this same
/// kernel, so sequential and parallel runs agree exactly.
#[inline]
fn sq_dist<F: Float>(a: &[F], b: &[F]) -> F {
    let mut acc = [F::ZERO; 8];
    let (chunks_a, rest_a) = a.as_chunks::<8>();
    let (chunks_b, rest_b) = b.as_chunks::<8>();
    for (xa, xb) in chunks_a.iter().zip(chunks_b) {
        for l in 0..8 {
            let d = xa[l] - xb[l];
            acc[l] = acc[l] + d * d;
        }
    }
    let mut s = F::ZERO;
    for v in acc {
        s = s + v;
    }
    for (&p, &q) in rest_a.iter().zip(rest_b) {
        let d = p - q;
        s = s + d * d;
    }
    s
}

/// Nearest centre to `row` (ties go to the lowest index), and its squared distance.
fn nearest_row<F: Float>(row: &[F], centers: &[F], k: usize, cols: usize) -> (usize, F) {
    let mut best = (0, sq_dist(row, &centers[..cols]));
    for c in 1..k {
        let d = sq_dist(row, &centers[c * cols..(c + 1) * cols]);
        if d < best.1 {
            best = (c, d);
        }
    }
    best
}

/// Calls `f(r, row)` for every sample. A contiguous view is used in place. Other
/// views have each row copied into one reused buffer.
fn for_each_row<F: Float, X: ArrayView<Elem = F>>(x: &X, mut f: impl FnMut(usize, &[F])) {
    let mut buf: Vec<F> = Vec::new();
    for r in 0..x.rows() {
        match x.row_slice(r) {
            Some(row) => f(r, row),
            None => {
                buf.clear();
                for j in 0..x.cols() {
                    buf.push(x.get(r, j));
                }
                f(r, &buf);
            }
        }
    }
}

fn copy_row<F: Float, X: ArrayView<Elem = F>>(x: &X, r: usize) -> Vec<F> {
    (0..x.cols()).map(|j| x.get(r, j)).collect()
}

/// A row-major copy of the data, made once per fit, and only when the parallel path
/// needs plain slices. Sequential runs never copy.
fn dense_copy_if_parallel<F: Float, X: ArrayView<Elem = F>>(
    x: &X,
    threads: usize,
) -> Option<Vec<F>> {
    #[cfg(feature = "parallel")]
    {
        if threads > 1 && x.rows() > 1 {
            let mut data = Vec::with_capacity(x.rows() * x.cols());
            for_each_row(x, |_, row| data.extend_from_slice(row));
            return Some(data);
        }
    }
    let _ = (x, threads);
    None
}

/// Assignment step: labels and total inertia. Inertia is summed in row order, so
/// it does not depend on how the work is split across threads.
fn assign<F: Float, X: ArrayView<Elem = F>>(
    x: &X,
    dense: Option<&[F]>,
    centers: &[F],
    k: usize,
    cols: usize,
    threads: usize,
) -> (Vec<usize>, F) {
    #[cfg(feature = "parallel")]
    {
        if let Some(data) = dense {
            if threads > 1 && x.rows() > 1 {
                return assign_parallel(data, x.rows(), centers, k, cols, threads);
            }
        }
    }
    let _ = (dense, threads);
    let mut labels = Vec::with_capacity(x.rows());
    let mut inertia = F::ZERO;
    for_each_row(x, |_, row| {
        let (c, d) = nearest_row(row, centers, k, cols);
        labels.push(c);
        inertia = inertia + d;
    });
    (labels, inertia)
}

#[cfg(feature = "parallel")]
fn assign_parallel<F: Float>(
    data: &[F],
    rows: usize,
    centers: &[F],
    k: usize,
    cols: usize,
    threads: usize,
) -> (Vec<usize>, F) {
    let chunk = rows.div_ceil(threads.min(rows)).max(1);
    let parts: Vec<Vec<(usize, F)>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..rows)
            .step_by(chunk)
            .map(|start| {
                let end = (start + chunk).min(rows);
                s.spawn(move || {
                    (start..end)
                        .map(|r| nearest_row(&data[r * cols..(r + 1) * cols], centers, k, cols))
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

/// Per-cluster coordinate ranges of the members, collected during the update.
struct Extent<F: Float> {
    lo: Vec<F>,
    hi: Vec<F>,
    populated: Vec<bool>,
}

/// Update step: each centre becomes the mean of its members. An empty cluster
/// keeps its previous centre. Also records each cluster's coordinate range.
fn update_centers<F: Float, X: ArrayView<Elem = F>>(
    x: &X,
    dense: Option<&[F]>,
    labels: &[usize],
    k: usize,
    cols: usize,
    centers: &mut [F],
    threads: usize,
) -> Extent<F> {
    let rows = x.rows();
    let chunks = rows.div_ceil(UPDATE_CHUNK);
    let partials = chunk_partials(x, dense, labels, k, cols, chunks, threads);

    // Merge in chunk order, so the sums do not depend on the thread count.
    let mut sums = vec![F::ZERO; k * cols];
    let mut counts = vec![0_usize; k];
    let mut lo = vec![F::ZERO; k * cols];
    let mut hi = vec![F::ZERO; k * cols];
    for p in &partials {
        for c in 0..k {
            if p.counts[c] == 0 {
                continue;
            }
            let fresh = counts[c] == 0;
            counts[c] += p.counts[c];
            for j in 0..cols {
                let idx = c * cols + j;
                sums[idx] = sums[idx] + p.sums[idx];
                if fresh || p.lo[idx] < lo[idx] {
                    lo[idx] = p.lo[idx];
                }
                if fresh || p.hi[idx] > hi[idx] {
                    hi[idx] = p.hi[idx];
                }
            }
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
    Extent {
        lo,
        hi,
        populated: counts.iter().map(|&n| n > 0).collect(),
    }
}

/// Rows per update chunk. Fixed, so the partial sums are the same for any thread
/// count.
const UPDATE_CHUNK: usize = 4096;

/// Per-cluster sums, counts and coordinate ranges over one chunk of rows.
struct Partial<F: Float> {
    sums: Vec<F>,
    counts: Vec<usize>,
    lo: Vec<F>,
    hi: Vec<F>,
}

impl<F: Float> Partial<F> {
    fn new(k: usize, cols: usize) -> Self {
        Self {
            sums: vec![F::ZERO; k * cols],
            counts: vec![0; k],
            lo: vec![F::ZERO; k * cols],
            hi: vec![F::ZERO; k * cols],
        }
    }

    fn add(&mut self, c: usize, row: &[F]) {
        let cols = row.len();
        let base = c * cols;
        if self.counts[c] == 0 {
            // First member: the range starts at this row.
            self.lo[base..base + cols].copy_from_slice(row);
            self.hi[base..base + cols].copy_from_slice(row);
        }
        self.counts[c] += 1;
        // Slice iterators with branch-free min and max, so the loop vectorises.
        let sums = &mut self.sums[base..base + cols];
        let lo = &mut self.lo[base..base + cols];
        let hi = &mut self.hi[base..base + cols];
        for (((s, l), h), &v) in sums
            .iter_mut()
            .zip(lo.iter_mut())
            .zip(hi.iter_mut())
            .zip(row)
        {
            *s = *s + v;
            *l = if v < *l { v } else { *l };
            *h = if v > *h { v } else { *h };
        }
    }
}

/// Partial statistics for chunk `chunk` of the rows.
fn chunk_partial<F: Float, X: ArrayView<Elem = F>>(
    x: &X,
    dense: Option<&[F]>,
    labels: &[usize],
    k: usize,
    cols: usize,
    chunk: usize,
) -> Partial<F> {
    let start = chunk * UPDATE_CHUNK;
    let end = (start + UPDATE_CHUNK).min(x.rows());
    let mut p = Partial::new(k, cols);
    match dense {
        Some(data) => {
            for (r, &c) in labels.iter().enumerate().take(end).skip(start) {
                p.add(c, &data[r * cols..(r + 1) * cols]);
            }
        }
        None => {
            let mut buf: Vec<F> = Vec::with_capacity(cols);
            for (r, &c) in labels.iter().enumerate().take(end).skip(start) {
                let row: &[F] = match x.row_slice(r) {
                    Some(s) => s,
                    None => {
                        buf.clear();
                        buf.extend((0..cols).map(|j| x.get(r, j)));
                        &buf
                    }
                };
                p.add(c, row);
            }
        }
    }
    p
}

/// All chunk partials, in chunk order.
fn chunk_partials<F: Float, X: ArrayView<Elem = F>>(
    x: &X,
    dense: Option<&[F]>,
    labels: &[usize],
    k: usize,
    cols: usize,
    chunks: usize,
    threads: usize,
) -> Vec<Partial<F>> {
    #[cfg(feature = "parallel")]
    {
        if let Some(data) = dense {
            if threads > 1 && chunks > 1 {
                // A plain number, not `x`: the generic view is not required to be `Sync`.
                let rows = x.rows();
                let workers = threads.min(chunks);
                let mut indexed: Vec<(usize, Partial<F>)> = std::thread::scope(|s| {
                    let handles: Vec<_> = (0..workers)
                        .map(|t| {
                            s.spawn(move || {
                                (t..chunks)
                                    .step_by(workers)
                                    .map(|c| {
                                        let start = c * UPDATE_CHUNK;
                                        let end = (start + UPDATE_CHUNK).min(rows);
                                        let mut p = Partial::new(k, cols);
                                        for r in start..end {
                                            p.add(labels[r], &data[r * cols..(r + 1) * cols]);
                                        }
                                        (c, p)
                                    })
                                    .collect::<Vec<_>>()
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .flat_map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
                        .collect()
                });
                indexed.sort_by_key(|(c, _)| *c);
                return indexed.into_iter().map(|(_, p)| p).collect();
            }
        }
    }
    let _ = threads;
    (0..chunks)
        .map(|c| chunk_partial(x, dense, labels, k, cols, c))
        .collect()
}

/// `true` if every populated cluster's centre lies within the per-coordinate range
/// of its members, to within rounding. A centre that is the mean of its members is
/// a convex combination of them, so this is a necessary condition for lying in their
/// convex hull. This is the check run for `kmeans.centers_in_hull`.
fn within_extent<F: Float>(centers: &[F], extent: &Extent<F>, k: usize, cols: usize) -> bool {
    for c in 0..k {
        if !extent.populated[c] {
            continue;
        }
        for j in 0..cols {
            let idx = c * cols + j;
            let (lo, hi) = (extent.lo[idx], extent.hi[idx]);
            let slack = F::EPSILON * F::from_usize(16) * (F::ONE + abs_f(lo) + abs_f(hi));
            let v = centers[idx];
            if v < lo - slack || v > hi + slack {
                return false;
            }
        }
    }
    true
}

fn abs_f<F: Float>(x: F) -> F {
    if x < F::ZERO {
        -x
    } else {
        x
    }
}

/// Rounding slack for the monotone-descent check. The contract is stated over the
/// reals, and the floating-point sums can drift by a few ulps.
fn descent_slack<F: Float>(scale: F) -> F {
    F::EPSILON * F::from_usize(64) * (F::ONE + scale)
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
    centers.extend(copy_row(x, first));

    // Weights are squared distances to the nearest chosen centre, in f64 for sampling.
    let mut weights = vec![0.0_f64; rows];
    for_each_row(x, |r, row| {
        weights[r] = sq_dist(row, &centers[0..cols]).to_f64()
    });
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
        centers.extend(copy_row(x, pick));
        let centre = centers[centers.len() - cols..].to_vec();
        for_each_row(x, |r, row| {
            let d = sq_dist(row, &centre).to_f64();
            if d < weights[r] {
                weights[r] = d;
            }
        });
    }
    centers
}

/// Random seeding: `k` distinct samples, chosen by a partial Fisher-Yates shuffle.
fn random_init<F: Float, X: ArrayView<Elem = F>>(x: &X, k: usize, rng: &mut SplitMix64) -> Vec<F> {
    let rows = x.rows();
    let mut order: Vec<usize> = (0..rows).collect();
    let mut centers = Vec::with_capacity(k * x.cols());
    for i in 0..k {
        let j = i + rng.next_below(rows - i);
        order.swap(i, j);
        centers.extend(copy_row(x, order[i]));
    }
    centers
}
