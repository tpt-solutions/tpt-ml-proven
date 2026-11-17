// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

//! Cyclic Jacobi eigendecomposition of a real symmetric matrix, in `f64`.
//!
//! Jacobi is simple to verify line by line and is accurate for the small
//! covariance matrices PCA produces. It is not the fastest method, and a faster
//! tridiagonal solver is a later optimisation.

/// Eigenvalues and eigenvectors of a symmetric matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Eigen {
    /// Eigenvalues, sorted in descending order.
    pub values: Vec<f64>,
    /// `vectors[i]` is the unit eigenvector for `values[i]`.
    pub vectors: Vec<Vec<f64>>,
}

/// Decomposes a symmetric `n x n` matrix given in row-major order.
///
/// `a` must be symmetric. Only its values are used, and the input is not
/// modified.
pub fn symmetric_eigen(a: &[f64], n: usize) -> Eigen {
    assert_eq!(a.len(), n * n, "matrix must be n x n");
    let mut m = a.to_vec();
    // v holds the accumulated rotations; its columns become the eigenvectors.
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let frob2: f64 = m.iter().map(|x| x * x).sum();
    for _sweep in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|p| (p + 1..n).map(move |q| (p, q)))
            .map(|(p, q)| m[p * n + q] * m[p * n + q])
            .sum();
        if off <= 1e-30 * frob2 || off == 0.0 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                rotate(&mut m, &mut v, n, p, q);
            }
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        m[j * n + j]
            .partial_cmp(&m[i * n + i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let values = order.iter().map(|&i| m[i * n + i]).collect();
    let vectors = order
        .iter()
        .map(|&col| (0..n).map(|row| v[row * n + col]).collect())
        .collect();
    Eigen { values, vectors }
}

/// One Jacobi rotation that zeroes `m[p][q]`.
fn rotate(m: &mut [f64], v: &mut [f64], n: usize, p: usize, q: usize) {
    let apq = m[p * n + q];
    if apq == 0.0 {
        return;
    }
    let app = m[p * n + p];
    let aqq = m[q * n + q];
    let theta = (aqq - app) / (2.0 * apq);
    let t = if theta >= 0.0 {
        1.0 / (theta + (theta * theta + 1.0).sqrt())
    } else {
        -1.0 / (-theta + (theta * theta + 1.0).sqrt())
    };
    let c = 1.0 / (t * t + 1.0).sqrt();
    let s = t * c;

    for r in 0..n {
        if r != p && r != q {
            let arp = m[r * n + p];
            let arq = m[r * n + q];
            let new_rp = c * arp - s * arq;
            let new_rq = c * arq + s * arp;
            m[r * n + p] = new_rp;
            m[p * n + r] = new_rp;
            m[r * n + q] = new_rq;
            m[q * n + r] = new_rq;
        }
    }
    m[p * n + p] = app - t * apq;
    m[q * n + q] = aqq + t * apq;
    m[p * n + q] = 0.0;
    m[q * n + p] = 0.0;

    for r in 0..n {
        let vrp = v[r * n + p];
        let vrq = v[r * n + q];
        v[r * n + p] = c * vrp - s * vrq;
        v[r * n + q] = c * vrq + s * vrp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matvec(a: &[f64], n: usize, x: &[f64]) -> Vec<f64> {
        (0..n)
            .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum())
            .collect()
    }

    #[test]
    fn diagonal_matrix_is_its_own_decomposition() {
        let a = [3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        let e = symmetric_eigen(&a, 3);
        assert_eq!(e.values, vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn two_by_two_known_eigenpairs() {
        // [[2, 1], [1, 2]] has eigenvalues 3 and 1.
        let a = [2.0, 1.0, 1.0, 2.0];
        let e = symmetric_eigen(&a, 2);
        assert!((e.values[0] - 3.0).abs() < 1e-12);
        assert!((e.values[1] - 1.0).abs() < 1e-12);
        let inv = 1.0 / 2.0_f64.sqrt();
        assert!((e.vectors[0][0].abs() - inv).abs() < 1e-12);
        assert!((e.vectors[0][1].abs() - inv).abs() < 1e-12);
    }

    #[test]
    fn residual_and_orthonormality_on_a_random_symmetric_matrix() {
        let n = 6;
        let mut seed = 12345_u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let v = next();
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
        let e = symmetric_eigen(&a, n);
        for (lambda, x) in e.values.iter().zip(&e.vectors) {
            let ax = matvec(&a, n, x);
            for (axi, xi) in ax.iter().zip(x) {
                assert!((axi - lambda * xi).abs() < 1e-10);
            }
            let norm: f64 = x.iter().map(|v| v * v).sum();
            assert!((norm - 1.0).abs() < 1e-10);
        }
        for i in 0..n {
            for j in i + 1..n {
                let dot: f64 = e.vectors[i]
                    .iter()
                    .zip(&e.vectors[j])
                    .map(|(a, b)| a * b)
                    .sum();
                assert!(dot.abs() < 1e-10);
            }
        }
    }

    #[test]
    fn zero_matrix_gives_zero_eigenvalues() {
        let e = symmetric_eigen(&[0.0; 4], 2);
        assert_eq!(e.values, vec![0.0, 0.0]);
    }
}
