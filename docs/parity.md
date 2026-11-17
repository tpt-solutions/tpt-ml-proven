# scikit-learn parity

The parity tests check that KMeans and PCA agree with scikit-learn to 1e-9. They
compare the numbers, not the code.

## What is compared

| Algorithm | Compared | Tolerance |
| --- | --- | --- |
| KMeans (Lloyd) | final centres, labels (up to relabelling), inertia | 1e-9, absolute for values near zero and relative otherwise |
| PCA (`svd_solver="full"`) | mean, explained variance, explained variance ratio, components | 1e-9 with the same scaling |

The reference values are in `tools/parity/fixtures/*.txt`. They come from
`tools/parity/generate_fixtures.py` with scikit-learn **1.9.0**. The script
refuses to run against any other version. CI never runs scikit-learn: the
fixtures are committed.

## Differences we normalise

These are real differences between the implementations. Each one is handled in the
test, and none is hidden.

1. **Starting centres.** k-means++ and random initialisation draw from different
   random number generators, so the two sides would start from different points.
   The fixtures give both sides the same explicit starting centres
   (`KMeans::with_initial_centers`), so the test compares the Lloyd iteration
   itself. k-means++ and random initialisation are not compared against
   scikit-learn. They are covered by the crate's own tests, for determinism and
   for the edge cases.

2. **Sign of principal axes.** The sign of an eigenvector is not fixed by the
   method. scikit-learn flips by the left singular vectors, and this crate flips so
   the largest-magnitude entry is positive. The PCA test flips each reference
   component to agree in sign with ours before comparing. Nothing else about the
   components is changed.

3. **Empty clusters.** scikit-learn moves an empty cluster's centre to a far point.
   This crate keeps the old centre, which the contract needs for monotone descent
   (see `kmeans.rs`). The generator refuses to write a KMeans fixture that leaves a
   cluster empty, so the test never reaches this case.

4. **Convergence stopping rule.** The test runs both sides with `tol = 0` until the
   labels stop changing, so both stop at the same fixed point.

## What is not compared

- **k-means++ and random initialisation** (see 1).
- **`n_iter`**. The counts are not part of the parity claim, so the benchmark reports
  them without asserting equality.
- **Floating-point summation order.** scikit-learn sums in its own order, so
  results can differ in the last bits. The 1e-9 tolerance absorbs that.
- **Other `svd_solver` settings** and whitening. Only the full decomposition is
  covered.

## Running

```sh
python tools/parity/generate_fixtures.py   # regenerate, only when the pin changes
cargo test -p tpt-ml-proven-clustering --test parity
cargo test -p tpt-ml-proven-decomposition --test parity
```
