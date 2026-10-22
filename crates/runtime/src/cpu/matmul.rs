//! The tiled matmul of the fast CPU backend, bit-identical to the reference.
//! It has its own module because code placement changed its speed by 9 percent
//! when other kernels shared the module (docs/benchmarks.md).
//!
//! The micro-kernel is written with explicit vectors (`fearless_simd`) so that
//! register use does not depend on what the autovectorizer makes of the loop,
//! and so that x86 picks AVX2 with FMA at run time without build flags. Every
//! multiply-add is `mul_add_precise`: one rounding on every level, in hardware
//! where it exists and in software where it does not. The cheaper `mul_add`
//! may round twice on levels without FMA, which would change the bits.

use fearless_simd::{Level, Simd, dispatch, f32x16, prelude::*};
use fearless_simd_macros::simd;

/// Cache tile sizes. `KC` rows of B by `NC` columns stay hot while every row
/// of A streams past them.
pub(super) const KC: usize = 256;
pub(super) const NC: usize = 512;
/// Rows and columns of C held in registers by the micro-kernel.
pub(super) const R: usize = 6;
pub(super) const W: usize = 16;

/// `c[r][..W] = fma(a[r][p], b[p][..W], c[r][..W])` for `p` in `0..klen`, for
/// `RR` rows. `a`, `b` and `c` start at the block; `lda`, `ldb` and `ldc` are
/// their row strides. An `RR x W` block of C lives in registers for the whole
/// run instead of being loaded and stored per step. The adds into each element
/// are the same, in the same order.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn micro<S: Simd, const RR: usize>(
    simd: S,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: &mut [f32],
    ldc: usize,
    klen: usize,
) {
    let mut acc = [f32x16::splat(simd, 0.0); RR];
    for (r, row) in acc.iter_mut().enumerate() {
        *row = f32x16::from_slice(simd, &c[r * ldc..r * ldc + W]);
    }
    for p in 0..klen {
        let bv = f32x16::from_slice(simd, &b[p * ldb..p * ldb + W]);
        for (r, row) in acc.iter_mut().enumerate() {
            let x = f32x16::splat(simd, a[r * lda + p]);
            *row = x.mul_add_precise(bv, *row);
        }
    }
    for (r, row) in acc.iter().enumerate() {
        row.store_slice(&mut c[r * ldc..r * ldc + W]);
    }
}

/// `micro` for a row count only known at run time (`1 ..= R`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn micro_rows<S: Simd>(
    simd: S,
    rows: usize,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: &mut [f32],
    ldc: usize,
    klen: usize,
) {
    match rows {
        1 => micro::<S, 1>(simd, a, lda, b, ldb, c, ldc, klen),
        2 => micro::<S, 2>(simd, a, lda, b, ldb, c, ldc, klen),
        3 => micro::<S, 3>(simd, a, lda, b, ldb, c, ldc, klen),
        4 => micro::<S, 4>(simd, a, lda, b, ldb, c, ldc, klen),
        5 => micro::<S, 5>(simd, a, lda, b, ldb, c, ldc, klen),
        _ => micro::<S, 6>(simd, a, lda, b, ldb, c, ldc, klen),
    }
}

/// Row-major `[m, k] x [k, n]` into `c` (`m * n`, zero on entry), bit-identical
/// to the reference kernel.
///
/// The loop order is j-tile, k-tile, row block, k, j. For any one output
/// element the contributions still arrive with `k` ascending (k-tiles run in
/// order, and `p` ascends inside a tile), starting from `+0.0`, one fused
/// multiply-add each (no reassociation). That is exactly the reference's sum,
/// so the bits match.
///
/// Columns that do not fill a vector are handled by copying that tile of B,
/// zero padded, into a small buffer and running the same micro-kernel on a
/// padded copy of the C block. The padded lanes compute values nobody reads
/// (an infinity in A gives NaN there), and only the real columns are copied
/// back.
#[simd]
#[allow(clippy::too_many_arguments)]
fn matmul_rows_simd<S: Simd>(
    simd: S,
    a: &[f32],
    b: &[f32],
    k: usize,
    n: usize,
    c: &mut [f32],
    kc: usize,
    nc: usize,
) {
    let m = c.len() / n;
    let mut packed = Vec::new();
    for j0 in (0..n).step_by(nc) {
        let j1 = (j0 + nc).min(n);
        let tail = (j1 - j0) % W;
        let j_full = j1 - tail;
        for k0 in (0..k).step_by(kc) {
            let k1 = (k0 + kc).min(k);
            let klen = k1 - k0;
            if tail > 0 {
                packed.clear();
                packed.resize(klen * W, 0.0f32);
                for p in 0..klen {
                    let from = (k0 + p) * n + j_full;
                    packed[p * W..p * W + tail].copy_from_slice(&b[from..from + tail]);
                }
            }
            for i in (0..m).step_by(R) {
                let rows = (m - i).min(R);
                let a_block = &a[i * k + k0..];
                for j in (j0..j_full).step_by(W) {
                    micro_rows(
                        simd,
                        rows,
                        a_block,
                        k,
                        &b[k0 * n + j..],
                        n,
                        &mut c[i * n + j..],
                        n,
                        klen,
                    );
                }
                if tail > 0 {
                    let mut edge = [0.0f32; R * W];
                    for r in 0..rows {
                        let from = (i + r) * n + j_full;
                        edge[r * W..r * W + tail].copy_from_slice(&c[from..from + tail]);
                    }
                    micro_rows(simd, rows, a_block, k, &packed, W, &mut edge, W, klen);
                    for r in 0..rows {
                        let to = (i + r) * n + j_full;
                        c[to..to + tail].copy_from_slice(&edge[r * W..r * W + tail]);
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn matmul_rows(
    level: Level,
    a: &[f32],
    b: &[f32],
    k: usize,
    n: usize,
    c: &mut [f32],
    kc: usize,
    nc: usize,
) {
    dispatch!(level, simd => matmul_rows_simd(simd, a, b, k, n, c, kc, nc));
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
    matmul_tiled_at(Level::new(), a, b, (m, k, n), (kc, nc))
}

/// `matmul_tiled` on a chosen SIMD level, so tests can run every level the
/// machine has and not only the best one.
pub(super) fn matmul_tiled_at(
    level: Level,
    a: &[f32],
    b: &[f32],
    (m, k, n): (usize, usize, usize),
    (kc, nc): (usize, usize),
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    matmul_rows(level, a, b, k, n, &mut c, kc, nc);
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
    let level = Level::new();
    let mut c = vec![0.0f32; m * n];
    c.par_chunks_mut(ROWS_PER_TASK * n)
        .enumerate()
        .for_each(|(task, rows)| {
            let first = task * ROWS_PER_TASK;
            matmul_rows(level, &a[first * k..], b, k, n, rows, KC, NC);
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
