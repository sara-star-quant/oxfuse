//! Naive reference kernels. Obviously correct beats fast here: these are the
//! oracle the optimized backend is tested against.

use oxfuse_core::Op;

/// Applies `op`. `inputs` and `shapes` are parallel slices. The graph was
/// verified, so shapes are consistent with the op's shape rule.
pub(crate) fn apply(op: &Op, inputs: &[&[f32]], shapes: &[&[usize]]) -> Vec<f32> {
    match op {
        Op::Relu => relu(inputs[0]),
        Op::Sigmoid => sigmoid(inputs[0]),
        Op::Add => zip_scalar(inputs, shapes, |x, y| x + y),
        Op::Mul => zip_scalar(inputs, shapes, |x, y| x * y),
        Op::Reshape { .. } => inputs[0].to_vec(),
        Op::Transpose { perm } => transpose(inputs[0], shapes[0], perm),
        Op::MatMul => matmul(
            inputs[0],
            inputs[1],
            shapes[0][0],
            shapes[0][1],
            shapes[1][1],
        ),
    }
}

/// Elementwise binary op over equal shapes or with one scalar operand.
fn zip_scalar(inputs: &[&[f32]], shapes: &[&[usize]], f: impl Fn(f32, f32) -> f32) -> Vec<f32> {
    let (a, b) = (inputs[0], inputs[1]);
    if shapes[0].is_empty() && !shapes[1].is_empty() {
        b.iter().map(|&y| f(a[0], y)).collect()
    } else if shapes[1].is_empty() && !shapes[0].is_empty() {
        a.iter().map(|&x| f(x, b[0])).collect()
    } else {
        a.iter().zip(b).map(|(&x, &y)| f(x, y)).collect()
    }
}

/// `x > 0 ? x : 0`, with NaN propagating. `-0.0` maps to `+0.0`.
fn relu(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| relu_one(x)).collect()
}

pub(crate) fn relu_one(x: f32) -> f32 {
    if x.is_nan() || x > 0.0 { x } else { 0.0 }
}

/// `1 / (1 + exp(-x))` in `f32`. Large negative `x` makes `exp` overflow to
/// infinity and the result 0 without producing NaN; NaN propagates.
fn sigmoid(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| sigmoid_one(x)).collect()
}

pub(crate) fn sigmoid_one(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Row-major `[m, k] x [k, n]`. Each output element sums in ascending `k`
/// order in `f32`, one fused multiply-add (a single rounding) per step.
/// `f32::mul_add` is exact on every platform, in hardware or in software, so
/// the bits do not depend on the CPU. The fast backend must return these exact bits; this
/// kernel itself is held to the documented bound against the float64 oracle.
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc = a[i * k + p].mul_add(b[p * n + j], acc);
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Output dimension `i` is input dimension `perm[i]`. Walks the output in
/// row-major order with an odometer index and reads the input through
/// permuted strides.
fn transpose(input: &[f32], in_shape: &[usize], perm: &[usize]) -> Vec<f32> {
    let rank = in_shape.len();
    let mut in_strides = vec![1usize; rank];
    for d in (0..rank.saturating_sub(1)).rev() {
        in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
    }
    let out_shape: Vec<usize> = perm.iter().map(|&p| in_shape[p]).collect();
    let src_strides: Vec<usize> = perm.iter().map(|&p| in_strides[p]).collect();
    let mut out = Vec::with_capacity(input.len());
    let mut index = vec![0usize; rank];
    for _ in 0..input.len() {
        let offset: usize = index.iter().zip(&src_strides).map(|(i, s)| i * s).sum();
        out.push(input[offset]);
        for d in (0..rank).rev() {
            index[d] += 1;
            if index[d] < out_shape[d] {
                break;
            }
            index[d] = 0;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply2(op: Op, a: &[f32], a_shape: &[usize], b: &[f32], b_shape: &[usize]) -> Vec<f32> {
        apply(&op, &[a, b], &[a_shape, b_shape])
    }

    #[test]
    fn add_and_mul_on_equal_shapes() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [10.0, 20.0, 30.0, 40.0];
        assert_eq!(
            apply2(Op::Add, &a, &[2, 2], &b, &[2, 2]),
            [11.0, 22.0, 33.0, 44.0]
        );
        assert_eq!(
            apply2(Op::Mul, &a, &[2, 2], &b, &[2, 2]),
            [10.0, 40.0, 90.0, 160.0]
        );
    }

    #[test]
    fn scalar_on_either_side() {
        let t = [1.0, 2.0, 3.0, 4.0];
        let s = [10.0];
        assert_eq!(
            apply2(Op::Add, &s, &[], &t, &[2, 2]),
            [11.0, 12.0, 13.0, 14.0]
        );
        assert_eq!(
            apply2(Op::Add, &t, &[2, 2], &s, &[]),
            [11.0, 12.0, 13.0, 14.0]
        );
        assert_eq!(
            apply2(Op::Mul, &s, &[], &t, &[2, 2]),
            [10.0, 20.0, 30.0, 40.0]
        );
        assert_eq!(
            apply2(Op::Mul, &t, &[2, 2], &s, &[]),
            [10.0, 20.0, 30.0, 40.0]
        );
        assert_eq!(apply2(Op::Mul, &[3.0], &[], &[4.0], &[]), [12.0]);
    }

    #[test]
    fn nan_and_infinity_follow_ieee() {
        assert!(apply2(Op::Add, &[f32::NAN], &[], &[1.0], &[])[0].is_nan());
        assert!(apply2(Op::Mul, &[f32::INFINITY], &[], &[0.0], &[])[0].is_nan());
        assert_eq!(
            apply2(Op::Add, &[f32::INFINITY], &[], &[1.0], &[])[0],
            f32::INFINITY
        );
    }

    #[test]
    fn sigmoid_known_values() {
        let out = apply(&Op::Sigmoid, &[&[0.0, 100.0, -100.0, f32::NAN]], &[&[4]]);
        assert_eq!(out[0], 0.5);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], 0.0);
        assert!(out[3].is_nan());
    }

    #[test]
    fn sigmoid_of_large_negative_input_does_not_produce_nan() {
        let out = apply(&Op::Sigmoid, &[&[-1000.0, f32::NEG_INFINITY]], &[&[2]]);
        assert_eq!(out, [0.0, 0.0]);
    }

    #[test]
    fn sigmoid_stays_within_four_epsilon_of_the_f64_formula() {
        let xs: Vec<f32> = (-2000..=2000).map(|i| i as f32 * 0.01).collect();
        let got = apply(&Op::Sigmoid, &[&xs], &[&[xs.len()]]);
        let mut worst = 0.0f64;
        for (&x, &g) in xs.iter().zip(&got) {
            let want = 1.0 / (1.0 + (-f64::from(x)).exp());
            worst = worst.max((f64::from(g) - want).abs() / want);
        }
        let bound = 4.0 * f64::from(f32::EPSILON);
        assert!(
            worst <= bound,
            "worst relative error {worst:e} exceeds {bound:e}"
        );
    }

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
    }

    #[test]
    fn matmul_two_by_two_by_hand() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let out = apply(&Op::MatMul, &[&a, &b], &[&[2, 2], &[2, 2]]);
        assert_eq!(out, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_non_square_by_hand() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [1.0, 1.0, 1.0];
        let out = apply(&Op::MatMul, &[&a, &b], &[&[2, 3], &[3, 1]]);
        assert_eq!(out, [6.0, 15.0]);
    }

    #[test]
    fn matmul_on_small_integers_is_bit_exact() {
        let a: Vec<f32> = (0..12).map(|i| ((i * 7) % 9) as f32 - 4.0).collect();
        let b: Vec<f32> = (0..8).map(|i| ((i * 5) % 7) as f32 - 3.0).collect();
        let out = apply(&Op::MatMul, &[&a, &b], &[&[3, 4], &[4, 2]]);
        for i in 0..3 {
            for j in 0..2 {
                let want: i32 = (0..4)
                    .map(|p| (a[i * 4 + p] as i32) * (b[p * 2 + j] as i32))
                    .sum();
                assert_eq!(out[i * 2 + j], want as f32);
            }
        }
    }

    #[test]
    fn matmul_stays_within_the_error_bound_of_the_f64_result() {
        let (m, k, n) = (5, 7, 3);
        let mut seed = 42u64;
        let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed)).collect();
        let b: Vec<f32> = (0..k * n).map(|_| lcg(&mut seed)).collect();
        let out = apply(&Op::MatMul, &[&a, &b], &[&[m, k], &[k, n]]);
        for i in 0..m {
            for j in 0..n {
                let mut want = 0.0f64;
                let mut magnitude = 0.0f64;
                for p in 0..k {
                    let term = f64::from(a[i * k + p]) * f64::from(b[p * n + j]);
                    want += term;
                    magnitude += term.abs();
                }
                let bound = k as f64 * f64::from(f32::EPSILON) * magnitude;
                let err = (f64::from(out[i * n + j]) - want).abs();
                assert!(
                    err <= bound,
                    "[{i},{j}] error {err:e} exceeds bound {bound:e}"
                );
            }
        }
    }

    /// `x = 1 + 2^-12` squares to `1 + 2^-11 + 2^-24`, which rounds to `1 + 2^-11`.
    /// Adding the second product to the first with one rounding keeps the lost
    /// `2^-24`; a separate multiply and add loses it and returns zero.
    #[test]
    fn matmul_rounds_once_per_multiply_add() {
        let x = 1.0 + 2f32.powi(-12);
        let a = [1.0 + 2f32.powi(-11), x];
        let b = [1.0, -x];
        let out = apply(&Op::MatMul, &[&a, &b], &[&[1, 2], &[2, 1]]);
        assert_eq!(out[0], -(2f32.powi(-24)));
    }

    #[test]
    fn matmul_propagates_nan() {
        let a = [f32::NAN, 1.0];
        let b = [1.0, 1.0];
        let out = apply(&Op::MatMul, &[&a, &b], &[&[1, 2], &[2, 1]]);
        assert!(out[0].is_nan());
    }

    #[test]
    fn reshape_copies_the_data_unchanged() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let op = Op::Reshape { shape: vec![3, 2] };
        assert_eq!(apply(&op, &[&data], &[&[2, 3]]), data);
    }

    #[test]
    fn transpose_two_by_three_by_hand() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let op = Op::Transpose { perm: vec![1, 0] };
        assert_eq!(
            apply(&op, &[&data], &[&[2, 3]]),
            [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn transpose_three_dimensions_by_hand() {
        // out[a][b][c] = in[b][c][a] for perm [2, 0, 1]; input shape [1, 2, 3].
        let data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let op = Op::Transpose {
            perm: vec![2, 0, 1],
        };
        assert_eq!(
            apply(&op, &[&data], &[&[1, 2, 3]]),
            [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
        );
    }

    #[test]
    fn transpose_handles_scalars_and_vectors() {
        let scalar = Op::Transpose { perm: vec![] };
        assert_eq!(apply(&scalar, &[&[7.0]], &[&[]]), [7.0]);
        let vector = Op::Transpose { perm: vec![0] };
        assert_eq!(
            apply(&vector, &[&[1.0, 2.0, 3.0]], &[&[3]]),
            [1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn transposing_twice_with_inverse_permutations_is_the_identity() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let forward = Op::Transpose {
            perm: vec![2, 0, 1],
        };
        let back = Op::Transpose {
            perm: vec![1, 2, 0],
        };
        let once = apply(&forward, &[&data], &[&[2, 3, 4]]);
        let twice = apply(&back, &[&once], &[&[4, 2, 3]]);
        assert_eq!(twice, data);
    }
}
