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
fn micro<S: Simd, const RR: usize, const NS: usize>(
    simd: S,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: &mut [f32],
    ldc: usize,
    klen: usize,
) {
    let mut acc = [[f32x16::splat(simd, 0.0); NS]; RR];
    for (r, row) in acc.iter_mut().enumerate() {
        for (s, v) in row.iter_mut().enumerate() {
            let at = r * ldc + s * W;
            *v = f32x16::from_slice(simd, &c[at..at + W]);
        }
    }
    for p in 0..klen {
        let mut bv = [f32x16::splat(simd, 0.0); NS];
        for (s, v) in bv.iter_mut().enumerate() {
            let at = p * ldb + s * W;
            *v = f32x16::from_slice(simd, &b[at..at + W]);
        }
        for (r, row) in acc.iter_mut().enumerate() {
            let x = f32x16::splat(simd, a[r * lda + p]);
            for (s, v) in row.iter_mut().enumerate() {
                *v = x.mul_add_precise(bv[s], *v);
            }
        }
    }
    for (r, row) in acc.iter().enumerate() {
        for (s, v) in row.iter().enumerate() {
            let at = r * ldc + s * W;
            v.store_slice(&mut c[at..at + W]);
        }
    }
}

/// `micro` for a row count only known at run time (`1 ..= R`), `ns` strips
/// of `W` columns wide (`ns` is 1 unless `wide_strips(rows)` says more).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn micro_rows<S: Simd>(
    simd: S,
    rows: usize,
    ns: usize,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: &mut [f32],
    ldc: usize,
    klen: usize,
) {
    match (rows, ns) {
        (1, 4) => micro::<S, 1, 4>(simd, a, lda, b, ldb, c, ldc, klen),
        (2, 2) => micro::<S, 2, 2>(simd, a, lda, b, ldb, c, ldc, klen),
        (1, _) => micro::<S, 1, 1>(simd, a, lda, b, ldb, c, ldc, klen),
        (2, _) => micro::<S, 2, 1>(simd, a, lda, b, ldb, c, ldc, klen),
        (3, _) => micro::<S, 3, 1>(simd, a, lda, b, ldb, c, ldc, klen),
        (4, _) => micro::<S, 4, 1>(simd, a, lda, b, ldb, c, ldc, klen),
        (5, _) => micro::<S, 5, 1>(simd, a, lda, b, ldb, c, ldc, klen),
        _ => micro::<S, 6, 1>(simd, a, lda, b, ldb, c, ldc, klen),
    }
}

/// Strips of `W` columns the next micro-kernel call covers, with `left` columns
/// (a multiple of `W`) remaining before the tail of the column tile. With one or two rows there are too few
/// independent accumulators to hide the latency of the fused multiply-add, so
/// the block is made wider instead of taller. The sums are the same, element by
/// element; only how many columns share one pass over `k` changes.
fn strips_for(rows: usize, left: usize) -> usize {
    let strips = left / W;
    let wide = match rows {
        1 => 4,
        2 => 2,
        _ => 1,
    };
    if strips >= wide { wide } else { 1 }
}

/// Of the `klen` rows of B from row `k0`, how many have a full vector of `W`
/// floats at column `j_full` inside a `b` of `b_len` floats with row length
/// `n`. The rest need the padded copy. Rows are in order, so it is a prefix.
fn direct_rows(b_len: usize, n: usize, j_full: usize, k0: usize, klen: usize) -> usize {
    if b_len < j_full + W {
        return 0;
    }
    ((b_len - j_full - W) / n + 1).saturating_sub(k0).min(klen)
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
/// Columns that do not fill a vector run through the same micro-kernel on a
/// padded copy of the C block. The kernel reads a full vector of B from the
/// first tail column, which runs on into the next row of B: those lanes
/// compute values nobody reads (an infinity in A gives NaN there), and only
/// the real columns are copied back. Only the last few rows of B, where a full
/// vector would read past the end of the slice, are copied into a zero padded
/// buffer first; they come last in `k`, so the order of the sums is unchanged.
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
            // Rows `p` of this k-tile whose vector at the first tail column
            // fits in `b`; the rest go through `packed`.
            let direct = if tail == 0 {
                klen
            } else {
                direct_rows(b.len(), n, j_full, k0, klen)
            };
            // Empty when every row is direct.
            packed.clear();
            for p in direct..klen {
                let from = (k0 + p) * n + j_full;
                let mut row = [0.0f32; W];
                row[..tail].copy_from_slice(&b[from..from + tail]);
                packed.extend_from_slice(&row);
            }
            for i in (0..m).step_by(R) {
                let rows = (m - i).min(R);
                let a_block = &a[i * k + k0..];
                let mut j = j0;
                while j < j_full {
                    let ns = strips_for(rows, j_full - j);
                    micro_rows(
                        simd,
                        rows,
                        ns,
                        a_block,
                        k,
                        &b[k0 * n + j..],
                        n,
                        &mut c[i * n + j..],
                        n,
                        klen,
                    );
                    j += ns * W;
                }
            }
            for i in (0..m).step_by(R) {
                let rows = (m - i).min(R);
                let a_block = &a[i * k + k0..];
                if tail > 0 {
                    let mut edge = [0.0f32; R * W];
                    for r in 0..rows {
                        let from = (i + r) * n + j_full;
                        edge[r * W..r * W + tail].copy_from_slice(&c[from..from + tail]);
                    }
                    // Either call does nothing when its row count is 0.
                    let b_tail = &b[k0 * n + j_full..];
                    micro_rows(simd, rows, 1, a_block, k, b_tail, n, &mut edge, W, direct);
                    let a_rest = &a[i * k + k0 + direct..];
                    let rest = klen - direct;
                    micro_rows(simd, rows, 1, a_rest, k, &packed, W, &mut edge, W, rest);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_rows_counts_the_rows_whose_vector_fits() {
        // 5 rows of 10: row p fits while p * 10 + 16 <= 50, so p = 0 to 3.
        assert_eq!(direct_rows(50, 10, 0, 0, 5), 4);
        // The same rows seen from the third one, and from a short tile.
        assert_eq!(direct_rows(50, 10, 0, 2, 3), 2);
        assert_eq!(direct_rows(50, 10, 0, 0, 2), 2);
        // A vector that ends exactly at the end of `b` fits.
        assert_eq!(direct_rows(W, W, 0, 0, 1), 1);
        // One float short does not.
        assert_eq!(direct_rows(W - 1, W, 0, 0, 1), 0);
        // A tail column further right leaves less room.
        assert_eq!(direct_rows(50, 20, 8, 0, 2), 2);
        assert_eq!(direct_rows(44, 20, 8, 0, 2), 2);
        assert_eq!(direct_rows(43, 20, 8, 0, 2), 1);
        // A tile that starts past the fitting rows has none.
        assert_eq!(direct_rows(50, 10, 0, 4, 1), 0);
    }
    #[test]
    fn strips_for_widens_only_a_whole_group_of_few_rows() {
        // One row: 4 strips when 4 or more remain, else one at a time.
        assert_eq!(strips_for(1, 4 * W), 4);
        assert_eq!(strips_for(1, 9 * W), 4);
        assert_eq!(strips_for(1, 3 * W), 1);
        assert_eq!(strips_for(1, W), 1);
        // Two rows: 2 strips when 2 or more remain.
        assert_eq!(strips_for(2, 2 * W), 2);
        assert_eq!(strips_for(2, 7 * W), 2);
        assert_eq!(strips_for(2, W), 1);
        // Three or more rows never widen.
        for rows in 3..=R {
            assert_eq!(strips_for(rows, 9 * W), 1);
        }
    }

    /// With `parallel`, the 64- and 70-row runs are split across threads (70 leaves
    /// a short last task) and the single rows are not, so equal bits also prove
    /// the thread split changes nothing.
    #[test]
    fn rows_of_a_batch_equal_the_single_row_runs_across_the_thread_threshold() {
        let (k, n) = (300, 118);
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i * 37 % 1999) as f32) / 700.0 - 1.4)
            .collect();
        for m in (1..=13).chain([31, 32, 33, 64, 70]) {
            let a: Vec<f32> = (0..m * k)
                .map(|i| ((i * 53 % 2003) as f32) / 600.0 - 1.6)
                .collect();
            let full = matmul(&a, &b, m, k, n);
            for r in 0..m {
                let one = matmul(&a[r * k..(r + 1) * k], &b, 1, k, n);
                let same = full[r * n..(r + 1) * n]
                    .iter()
                    .zip(&one)
                    .all(|(x, y)| x.to_bits() == y.to_bits());
                assert!(same, "row {r} of {m} differs from the row alone");
            }
        }
        #[cfg(feature = "parallel")]
        {
            assert!(use_threads(64, k, n), "the 64-row run must use threads");
            assert!(!use_threads(1, k, n), "a single row must not");
        }
    }
}
