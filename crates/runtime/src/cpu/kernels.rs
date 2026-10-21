//! Kernels of the fast CPU backend. Every kernel must return exactly the bits
//! the reference kernel returns. An arm delegates to the reference kernel
//! until a benchmark shows an optimized version is faster.

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
        Op::Sigmoid => sigmoid(inputs[0]),
        Op::Relu | Op::Add | Op::Mul | Op::Reshape { .. } => {
            crate::kernels::apply(op, inputs, shapes)
        }
    }
}

/// Elements below which splitting Sigmoid across threads costs more than it
/// saves. Only Sigmoid is split: it is bound by `exp`, while Relu, Add and Mul
/// are bound by memory and got slower when split (see docs/benchmarks.md).
#[cfg(feature = "parallel")]
const SIGMOID_PARALLEL_MIN: usize = 1 << 16;
#[cfg(feature = "parallel")]
const SIGMOID_CHUNK: usize = 1 << 13;

/// Each element is computed alone, so the bits do not depend on the split.
fn sigmoid(input: &[f32]) -> Vec<f32> {
    #[cfg(feature = "parallel")]
    if input.len() >= SIGMOID_PARALLEL_MIN {
        use rayon::prelude::*;
        let mut out = vec![0.0f32; input.len()];
        out.par_chunks_mut(SIGMOID_CHUNK)
            .zip(input.par_chunks(SIGMOID_CHUNK))
            .for_each(|(out, input)| {
                for (o, &x) in out.iter_mut().zip(input) {
                    *o = crate::kernels::sigmoid_one(x);
                }
            });
        return out;
    }
    crate::kernels::apply(&Op::Sigmoid, &[input], &[])
}

fn sigmoid_in_place(buf: &mut [f32]) {
    #[cfg(feature = "parallel")]
    if buf.len() >= SIGMOID_PARALLEL_MIN {
        use rayon::prelude::*;
        buf.par_chunks_mut(SIGMOID_CHUNK).for_each(|chunk| {
            chunk
                .iter_mut()
                .for_each(|x| *x = crate::kernels::sigmoid_one(*x))
        });
        return;
    }
    buf.iter_mut()
        .for_each(|x| *x = crate::kernels::sigmoid_one(*x));
}

/// Computes an elementwise `op` into `buf`, which holds one operand and is
/// overwritten with the result. `other` is the second operand of Add and Mul
/// (one element for a scalar); `buf_first` says which operand `buf` held, so
/// the operation sees the operands in the same order as the reference.
pub(super) fn apply_in_place(op: &Op, buf: &mut [f32], other: Option<&[f32]>, buf_first: bool) {
    let zip = |buf: &mut [f32], other: &[f32], f: fn(f32, f32) -> f32| {
        let at = |i: usize| other[if other.len() == 1 { 0 } else { i }];
        for (i, v) in buf.iter_mut().enumerate() {
            *v = if buf_first {
                f(*v, at(i))
            } else {
                f(at(i), *v)
            };
        }
    };
    match op {
        Op::Relu => buf
            .iter_mut()
            .for_each(|x| *x = crate::kernels::relu_one(*x)),
        Op::Sigmoid => sigmoid_in_place(buf),
        Op::Add => zip(buf, other.expect("Add has two operands"), |x, y| x + y),
        Op::Mul => zip(buf, other.expect("Mul has two operands"), |x, y| x * y),
        Op::Reshape { .. } | Op::MatMul | Op::Transpose { .. } => {
            unreachable!("not an in-place op")
        }
    }
}

/// Cache tile sizes. `KC` rows of B by `NC` columns stay hot while every row
/// of A streams past them.
const KC: usize = 256;
const NC: usize = 512;
/// Rows and columns of C held in registers by the micro-kernel.
const R: usize = 4;
const W: usize = 16;

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

fn matmul_tiled(
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
const PARALLEL_MIN_MACS: usize = 1 << 21;

#[cfg(feature = "parallel")]
fn use_threads(m: usize, k: usize, n: usize) -> bool {
    m.saturating_mul(k).saturating_mul(n) >= PARALLEL_MIN_MACS
}

#[cfg(feature = "parallel")]
fn matmul_parallel(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    #[cfg(feature = "parallel")]
    if use_threads(m, k, n) {
        return matmul_parallel(a, b, m, k, n);
    }
    matmul_tiled(a, b, m, k, n, KC, NC)
}

const TILE: usize = 32;

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

/// `[rows, cols] -> [cols, rows]` in `TILE x TILE` blocks so both the reads
/// and the writes stay within a few cache lines.
fn transpose_2d(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for r0 in (0..rows).step_by(TILE) {
        for c0 in (0..cols).step_by(TILE) {
            for r in r0..(r0 + TILE).min(rows) {
                for c in c0..(c0 + TILE).min(cols) {
                    out[c * rows + r] = input[r * cols + c];
                }
            }
        }
    }
    out
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
        let sizes = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 64, 65];
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
        // Sizes that fill the 4 x W block, with leftover rows and columns.
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

    #[test]
    fn rank_two_transpose_crosses_tile_boundaries_correctly() {
        for (rows, cols) in [(1, 100), (100, 1), (31, 33), (32, 32), (33, 31), (65, 129)] {
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
