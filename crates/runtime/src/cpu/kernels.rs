//! Kernels of the fast CPU backend. Every kernel must return exactly the bits
//! the reference kernel returns. An arm delegates to the reference kernel
//! until a benchmark shows an optimized version is faster.

use super::matmul::matmul;
#[cfg(test)]
use super::matmul::{KC, NC, R, W, matmul_tiled};
#[cfg(all(test, feature = "parallel"))]
use super::matmul::{PARALLEL_MIN_MACS, matmul_parallel, use_threads};
use oxfuse_core::Op;

pub(super) fn apply(op: &Op, inputs: &[&[f32]], shapes: &[&[usize]]) -> Vec<f32> {
    match op {
        Op::MatMul => matmul(
            inputs[0],
            inputs[1],
            shapes[0][0],
            shapes[0][1],
            shapes[1][1],
        ),
        Op::Transpose { perm } => transpose(inputs[0], shapes[0], perm),
        Op::Sigmoid => super::elementwise::sigmoid(inputs[0]),
        Op::Relu | Op::Add | Op::Mul | Op::Reshape { .. } => {
            crate::kernels::apply(op, inputs, shapes)
        }
    }
}

const TILE: usize = 16;

/// Output dimension `i` is input dimension `perm[i]`; a pure copy, so the bits
/// are the input's bits in a different order.
fn transpose(input: &[f32], in_shape: &[usize], perm: &[usize]) -> Vec<f32> {
    let rank = in_shape.len();
    if rank < 2 || perm.iter().enumerate().all(|(i, &p)| i == p) {
        return input.to_vec();
    }
    if rank == 2 {
        return transpose_2d(input, in_shape[0], in_shape[1]);
    }
    let mut in_strides = vec![1usize; rank];
    for d in (0..rank - 1).rev() {
        in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
    }
    let out_shape: Vec<usize> = perm.iter().map(|&p| in_shape[p]).collect();
    let src_strides: Vec<usize> = perm.iter().map(|&p| in_strides[p]).collect();
    let (last_len, last_stride) = (out_shape[rank - 1], src_strides[rank - 1]);
    let mut out = Vec::with_capacity(input.len());
    // Odometer over every output dimension but the last; the last runs inline.
    let mut index = vec![0usize; rank - 1];
    let mut base = 0usize;
    loop {
        out.extend((0..last_len).map(|j| input[base + j * last_stride]));
        let mut d = rank - 1;
        loop {
            if d == 0 {
                return out;
            }
            d -= 1;
            index[d] += 1;
            base += src_strides[d];
            if index[d] < out_shape[d] {
                break;
            }
            base -= index[d] * src_strides[d];
            index[d] = 0;
        }
    }
}

/// `[rows, cols] -> [cols, rows]`. Each tile writes 16 contiguous floats of an
/// output row (one cache line) and reads them from 16 input rows; with 16
/// rows per tile this was twice as fast as 32 x 32 tiles at 1024 x 1024.
fn transpose_2d(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    #[cfg(feature = "parallel")]
    if input.len() >= TRANSPOSE_PARALLEL_MIN {
        use rayon::prelude::*;
        out.par_chunks_mut(TRANSPOSE_BAND * rows)
            .enumerate()
            .for_each(|(band, chunk)| {
                transpose_band(input, rows, cols, band * TRANSPOSE_BAND, chunk);
            });
        return out;
    }
    transpose_band(input, rows, cols, 0, &mut out);
    out
}

/// Elements below which threads cost more than they save.
#[cfg(feature = "parallel")]
const TRANSPOSE_PARALLEL_MIN: usize = 1 << 18;
/// Output rows per parallel task.
#[cfg(feature = "parallel")]
const TRANSPOSE_BAND: usize = 64;

/// Fills the output rows `c0 ..` that fit in `out`, a whole number of rows.
fn transpose_band(input: &[f32], rows: usize, cols: usize, c0: usize, out: &mut [f32]) {
    let c_end = c0 + out.len() / rows;
    for r0 in (0..rows).step_by(TILE) {
        let r1 = (r0 + TILE).min(rows);
        for t0 in (c0..c_end).step_by(TILE) {
            for c in t0..(t0 + TILE).min(c_end) {
                let dst = &mut out[(c - c0) * rows + r0..(c - c0) * rows + r1];
                let src = &input[r0 * cols + c..];
                for (i, d) in dst.iter_mut().enumerate() {
                    *d = src[i * cols];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        crate::kernels::apply(&Op::MatMul, &[a, b], &[&[m, k], &[k, n]])
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// Deterministic values in [-2, 2), non-dyadic so every sum rounds.
    fn values(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
            })
            .collect()
    }

    #[test]
    fn matmul_is_bit_identical_for_awkward_sizes() {
        let sizes = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 31, 33, 64, 65,
        ];
        for &m in &sizes {
            for &k in &sizes {
                for &n in &[1, 2, 5, 8, 9, 16, 17, 33, 40] {
                    let (a, b) = (values(m * k, 1), values(k * n, 2));
                    let want = bits(&reference_matmul(&a, &b, m, k, n));
                    // Tiny tiles force many tile boundaries on small inputs.
                    for (kc, nc) in [
                        (1, 1),
                        (3, 4),
                        (4, 3),
                        (8, 8),
                        (3, W),
                        (5, W + 1),
                        (2, 2 * W + 3),
                        (KC, NC),
                    ] {
                        assert_eq!(
                            bits(&matmul_tiled(&a, &b, m, k, n, kc, nc)),
                            want,
                            "m={m} k={k} n={n} kc={kc} nc={nc}"
                        );
                    }
                }
            }
        }
    }

    /// Rows below, at and above one and two register blocks, each with a full
    /// block of columns, a tail of columns and a k that spans two k-tiles.
    #[test]
    fn matmul_covers_every_row_remainder_of_the_register_block() {
        for m in 1..=2 * R + 1 {
            for n in [1, W - 1, W, W + 1, 2 * W + 3] {
                let k = 5;
                let (a, b) = (values(m * k, 31), values(k * n, 32));
                let want = bits(&reference_matmul(&a, &b, m, k, n));
                for (kc, nc) in [(2, W), (KC, NC)] {
                    assert_eq!(
                        bits(&matmul_tiled(&a, &b, m, k, n, kc, nc)),
                        want,
                        "m={m} n={n} kc={kc} nc={nc}"
                    );
                }
            }
        }
    }

    /// The multiply-add rounds once in the register block, in the column tail
    /// and in the row tail: the same lost `2^-24` as in the reference test.
    #[test]
    fn matmul_rounds_once_in_every_part_of_the_kernel() {
        let x = 1.0 + 2f32.powi(-12);
        for (m, n) in [
            (R, W),
            (R, W + 1),
            (R + 1, W),
            (1, 1),
            (2 * R + 1, 2 * W + 3),
        ] {
            let mut a = vec![0.0f32; m * 2];
            let mut b = vec![0.0f32; 2 * n];
            for i in 0..m {
                a[i * 2] = 1.0 + 2f32.powi(-11);
                a[i * 2 + 1] = x;
            }
            for j in 0..n {
                b[j] = 1.0;
                b[n + j] = -x;
            }
            for (kc, nc) in [(1, W), (2, 2 * W), (KC, NC)] {
                let out = matmul_tiled(&a, &b, m, 2, n, kc, nc);
                assert!(
                    out.iter().all(|&v| v == -(2f32.powi(-24))),
                    "m={m} n={n} kc={kc} nc={nc}: {out:?}"
                );
            }
        }
    }

    #[test]
    fn matmul_special_values_match_the_reference_bits() {
        let (m, k, n) = (3, 5, 4);
        let mut a = values(m * k, 3);
        let mut b = values(k * n, 4);
        a[1] = f32::NAN;
        a[7] = f32::INFINITY;
        a[8] = 0.0;
        b[8] = f32::NEG_INFINITY;
        b[3] = -0.0;
        a[12] = -0.0;
        let want = bits(&reference_matmul(&a, &b, m, k, n));
        for (kc, nc) in [(1, 1), (2, 3), (KC, NC)] {
            assert_eq!(
                bits(&matmul_tiled(&a, &b, m, k, n, kc, nc)),
                want,
                "kc={kc} nc={nc}"
            );
        }
    }

    #[test]
    fn a_zero_against_infinity_is_still_nan() {
        // 0 * inf = NaN. A kernel that skips zero operands would return 0.
        let out = matmul(&[0.0, 0.0], &[f32::INFINITY, 1.0], 1, 2, 1);
        assert!(out[0].is_nan());
    }

    #[test]
    fn special_values_inside_the_register_block_match_the_reference_bits() {
        // Sizes that fill the R x W block, with leftover rows and columns.
        let special = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            1.0e30,
            -1.0e-30,
        ];
        for (m, k, n) in [(4, 5, W), (5, 7, W + 1), (9, 6, 2 * W + 3), (8, 20, W)] {
            let mut a = values(m * k, 21);
            let mut b = values(k * n, 22);
            for (i, v) in a.iter_mut().enumerate().filter(|(i, _)| i % 5 == 2) {
                *v = special[i % special.len()];
            }
            for (i, v) in b.iter_mut().enumerate().filter(|(i, _)| i % 7 == 3) {
                *v = special[(i / 7) % special.len()];
            }
            let want = bits(&reference_matmul(&a, &b, m, k, n));
            for (kc, nc) in [(1, W), (3, W + 1), (KC, NC)] {
                assert_eq!(
                    bits(&matmul_tiled(&a, &b, m, k, n, kc, nc)),
                    want,
                    "m={m} k={k} n={n} kc={kc} nc={nc}"
                );
            }
        }
    }

    #[test]
    fn a_zero_against_infinity_is_still_nan_inside_the_register_block() {
        let (m, k, n) = (4, 3, W);
        let a = vec![0.0f32; m * k];
        let b = vec![f32::INFINITY; k * n];
        let out = matmul(&a, &b, m, k, n);
        assert!(out.iter().all(|v| v.is_nan()), "{out:?}");
    }

    #[test]
    fn the_public_entry_point_agrees_with_the_reference() {
        let (m, k, n) = (50, 70, 90);
        let (a, b) = (values(m * k, 5), values(k * n, 6));
        assert_eq!(
            bits(&matmul(&a, &b, m, k, n)),
            bits(&reference_matmul(&a, &b, m, k, n))
        );
    }

    fn reference_transpose(input: &[f32], shape: &[usize], perm: &[usize]) -> Vec<f32> {
        crate::kernels::apply(
            &Op::Transpose {
                perm: perm.to_vec(),
            },
            &[input],
            &[shape],
        )
    }

    fn permutations(n: usize) -> Vec<Vec<usize>> {
        if n == 0 {
            return vec![vec![]];
        }
        let mut all = Vec::new();
        for rest in permutations(n - 1) {
            for at in 0..=rest.len() {
                let mut p = rest.clone();
                p.insert(at, n - 1);
                all.push(p);
            }
        }
        all
    }

    #[test]
    fn transpose_is_bit_identical_for_every_permutation_and_awkward_shape() {
        let dims = [1usize, 2, 3, 5, 8, 9, 33];
        for rank in 0..=4 {
            for perm in permutations(rank) {
                for start in 0..dims.len() {
                    let shape: Vec<usize> = (0..rank)
                        .map(|i| dims[(start + 2 * i) % dims.len()])
                        .collect();
                    let len: usize = shape.iter().product();
                    let input = values(len, 7 + start as u64);
                    assert_eq!(
                        bits(&transpose(&input, &shape, &perm)),
                        bits(&reference_transpose(&input, &shape, &perm)),
                        "shape={shape:?} perm={perm:?}"
                    );
                }
            }
        }
    }

    /// Every shape up to 20 x 20 (a full tile of 16 and a tail), plus shapes
    /// around two tiles, and every band of output rows of each: each band
    /// element holds the matching input element. A Kani proof of the same
    /// statement did not finish in 15 minutes (the solver cannot bound the
    /// tile loops), so the bounded space is enumerated instead.
    #[test]
    fn transpose_band_places_every_element_for_every_shape_and_band() {
        let shapes = (1..=20).chain([31, 32, 33, 34, 47, 48, 49]);
        for rows in shapes.clone() {
            for cols in shapes.clone() {
                let input: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
                for c0 in 0..cols {
                    for count in 1..=cols - c0 {
                        let mut out = vec![-1.0f32; count * rows];
                        transpose_band(&input, rows, cols, c0, &mut out);
                        for c in c0..c0 + count {
                            for r in 0..rows {
                                assert_eq!(
                                    out[(c - c0) * rows + r],
                                    input[r * cols + c],
                                    "{rows}x{cols} band {c0}+{count} at ({r},{c})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rank_two_transpose_crosses_tile_boundaries_correctly() {
        for (rows, cols) in [
            (1, 100),
            (100, 1),
            (15, 17),
            (16, 16),
            (17, 15),
            (31, 33),
            (32, 32),
            (33, 31),
            (65, 129),
            // Large enough to be split into bands with the `parallel` feature,
            // with a short last band and a short last tile.
            (515, 513),
            (520, 512),
            (1, 300_000),
            (300_000, 1),
        ] {
            let input = values(rows * cols, 9);
            assert_eq!(
                bits(&transpose(&input, &[rows, cols], &[1, 0])),
                bits(&reference_transpose(&input, &[rows, cols], &[1, 0])),
                "{rows}x{cols}"
            );
        }
    }

    #[test]
    fn transpose_preserves_special_values() {
        let input = [f32::NAN, -0.0, f32::INFINITY, 0.0, 1.0, f32::NEG_INFINITY];
        assert_eq!(
            bits(&transpose(&input, &[2, 3], &[1, 0])),
            bits(&reference_transpose(&input, &[2, 3], &[1, 0]))
        );
    }

    #[cfg(feature = "parallel")]
    fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(f)
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn thread_count_does_not_change_the_bits() {
        let (m, k, n) = (96, 130, 70);
        let (a, b) = (values(m * k, 11), values(k * n, 12));
        let want = bits(&reference_matmul(&a, &b, m, k, n));
        for threads in [1, 2, 4, 7] {
            let got = in_pool(threads, || matmul_parallel(&a, &b, m, k, n));
            assert_eq!(bits(&got), want, "{threads} threads");
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_matmul_handles_awkward_row_counts() {
        for m in [1, 2, 15, 16, 17, 31, 33, 64, 65] {
            for (k, n) in [(9, 11), (13, 2 * W + 3)] {
                let (a, b) = (values(m * k, 13), values(k * n, 14));
                assert_eq!(
                    bits(&matmul_parallel(&a, &b, m, k, n)),
                    bits(&reference_matmul(&a, &b, m, k, n)),
                    "m={m} k={k} n={n}"
                );
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn the_thread_threshold_is_exact() {
        assert!(use_threads(1, PARALLEL_MIN_MACS, 1));
        assert!(!use_threads(1, PARALLEL_MIN_MACS - 1, 1));
        assert!(use_threads(PARALLEL_MIN_MACS, 1, 1));
        assert!(use_threads(1, 1, PARALLEL_MIN_MACS));
        assert!(!use_threads(0, usize::MAX, usize::MAX));
        assert!(use_threads(usize::MAX, usize::MAX, usize::MAX));
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn the_public_entry_point_agrees_with_the_reference_above_the_thread_threshold() {
        let n = 128;
        assert!(use_threads(n, n, n));
        let (a, b) = (values(n * n, 15), values(n * n, 16));
        assert_eq!(
            bits(&matmul(&a, &b, n, n, n)),
            bits(&reference_matmul(&a, &b, n, n, n))
        );
    }
}
