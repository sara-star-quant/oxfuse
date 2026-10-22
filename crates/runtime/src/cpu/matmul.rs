//! The tiled matmul of the fast CPU backend, bit-identical to the reference.
//! It has its own module because code placement changed its speed by 9 percent
//! when other kernels shared the module (docs/benchmarks.md).

/// Cache tile sizes. `KC` rows of B by `NC` columns stay hot while every row
/// of A streams past them.
pub(super) const KC: usize = 256;
pub(super) const NC: usize = 512;
/// Rows and columns of C held in registers by the micro-kernel.
const R: usize = 4;
pub(super) const W: usize = 16;

/// Row-major `[m, k] x [k, n]`, bit-identical to the reference kernel.
///
/// The loop order is j-tile, k-tile, row, k, j. For any one output element
/// the contributions still arrive with `k` ascending (k-tiles run in order,
/// and `p` ascends inside a tile), starting from `+0.0`, one multiply and one
/// add each (no fused multiply-add, no reassociation). That is exactly the
/// reference's sum, so the bits match. The innermost loop runs along a row of
/// B and a row of C, which the compiler turns into SIMD.
fn matmul_rows(a: &[f32], b: &[f32], k: usize, n: usize, c: &mut [f32], kc: usize, nc: usize) {
    let m = c.len() / n;
    for j0 in (0..n).step_by(nc) {
        let j1 = (j0 + nc).min(n);
        for k0 in (0..k).step_by(kc) {
            let k1 = (k0 + kc).min(k);
            let full = m - m % R;
            for i in (0..full).step_by(R) {
                let mut j = j0;
                while j + W <= j1 {
                    // A 4 x W block of C lives in registers for the whole
                    // k-tile instead of being loaded and stored per step. The
                    // adds into each element are the same, in the same order.
                    let mut acc = [[0.0f32; W]; R];
                    for (r, row) in acc.iter_mut().enumerate() {
                        row.copy_from_slice(&c[(i + r) * n + j..(i + r) * n + j + W]);
                    }
                    for p in k0..k1 {
                        let bv: &[f32; W] = b[p * n + j..p * n + j + W]
                            .try_into()
                            .expect("slice is W long");
                        for (r, row) in acc.iter_mut().enumerate() {
                            let x = a[(i + r) * k + p];
                            for (acc_j, &b_j) in row.iter_mut().zip(bv) {
                                *acc_j += x * b_j;
                            }
                        }
                    }
                    for (r, row) in acc.iter().enumerate() {
                        c[(i + r) * n + j..(i + r) * n + j + W].copy_from_slice(row);
                    }
                    j += W;
                }
                if j < j1 {
                    for r in 0..R {
                        let c_row = &mut c[(i + r) * n + j..(i + r) * n + j1];
                        for p in k0..k1 {
                            let a_ip = a[(i + r) * k + p];
                            let b_row = &b[p * n + j..p * n + j1];
                            for (c_ij, &b_pj) in c_row.iter_mut().zip(b_row) {
                                *c_ij += a_ip * b_pj;
                            }
                        }
                    }
                }
            }
            for i in full..m {
                let c_row = &mut c[i * n + j0..i * n + j1];
                for p in k0..k1 {
                    let a_ip = a[i * k + p];
                    let b_row = &b[p * n + j0..p * n + j1];
                    for (c_ij, &b_pj) in c_row.iter_mut().zip(b_row) {
                        *c_ij += a_ip * b_pj;
                    }
                }
            }
        }
    }
}

pub(super) fn matmul_tiled(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    kc: usize,
    nc: usize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    matmul_rows(a, b, k, n, &mut c, kc, nc);
    c
}

/// Rows per parallel task. Threads split the output by rows only; the sum
/// for each element is never split across threads, so the bits do not
/// depend on the thread count.
#[cfg(feature = "parallel")]
const ROWS_PER_TASK: usize = 16;

/// Multiply-adds below which threads cost more than they save. Set from the
/// benchmark crossover recorded in docs/benchmarks.md.
#[cfg(feature = "parallel")]
pub(super) const PARALLEL_MIN_MACS: usize = 1 << 21;

#[cfg(feature = "parallel")]
pub(super) fn use_threads(m: usize, k: usize, n: usize) -> bool {
    m.saturating_mul(k).saturating_mul(n) >= PARALLEL_MIN_MACS
}

#[cfg(feature = "parallel")]
pub(super) fn matmul_parallel(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let mut c = vec![0.0f32; m * n];
    c.par_chunks_mut(ROWS_PER_TASK * n)
        .enumerate()
        .for_each(|(task, rows)| {
            let first = task * ROWS_PER_TASK;
            matmul_rows(&a[first * k..], b, k, n, rows, KC, NC);
        });
    c
}

pub(super) fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    #[cfg(feature = "parallel")]
    if use_threads(m, k, n) {
        return matmul_parallel(a, b, m, k, n);
    }
    matmul_tiled(a, b, m, k, n, KC, NC)
}
