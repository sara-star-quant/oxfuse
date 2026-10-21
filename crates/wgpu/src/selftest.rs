//! A measurement of one device against the reference backend, in the same terms
//! as the contract: what is exact, what is bounded, what the device does with
//! the values the shading language does not pin down. Anyone with another
//! device can run it (`oxfuse gpu-selftest`) and attach the output to an issue.

use crate::bounds::{SIGMOID_ULP_BOUND, gpu_matmul_bound, sigmoid_matches, ulp_distance};
use crate::context::Context;
use crate::exec::WgpuBackend;
use oxfuse_core::{Error, GraphBuilder, Limits, Op};
use oxfuse_runtime::{Backend, ReferenceBackend, Tensor};
use std::fmt;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct SelfTestReport {
    pub adapter: String,
    pub api: String,
    pub limits: Vec<(&'static str, u64)>,
    /// Add, Mul, Transpose and Reshape on finite normal values equal the reference bit for bit.
    pub finite_tier_exact: bool,
    /// Relu equals the reference on a sweep over every exponent, both signs.
    pub relu_exact_on_all_sampled_patterns: bool,
    /// Every sampled MatMul element is within the documented bound of the f64 oracle.
    pub matmul_within_bound: bool,
    pub matmul_max_ulp: u32,
    pub sigmoid_max_ulp: u32,
    pub subnormal_add_flushed: bool,
    pub nan_in_add_gives: &'static str,
}

impl fmt::Display for SelfTestReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "adapter: {}", self.adapter)?;
        writeln!(f, "api: {}", self.api)?;
        writeln!(f, "finite tier exact: {}", self.finite_tier_exact)?;
        writeln!(
            f,
            "relu exact on sampled patterns: {}",
            self.relu_exact_on_all_sampled_patterns
        )?;
        writeln!(f, "matmul within bound: {}", self.matmul_within_bound)?;
        writeln!(f, "matmul max ulp from reference: {}", self.matmul_max_ulp)?;
        writeln!(
            f,
            "sigmoid max ulp from reference: {} (recorded bound on the first adapter: {SIGMOID_ULP_BOUND})",
            self.sigmoid_max_ulp
        )?;
        writeln!(f, "subnormal add flushed: {}", self.subnormal_add_flushed)?;
        writeln!(f, "nan in add gives: {}", self.nan_in_add_gives)?;
        for (name, value) in &self.limits {
            writeln!(f, "{name}: {value}")?;
        }
        Ok(())
    }
}

fn run_op(backend: &dyn Backend, op: Op, inputs: &[Tensor]) -> Result<Vec<f32>, Error> {
    let mut b = GraphBuilder::new(Limits::default());
    let ids = inputs
        .iter()
        .enumerate()
        .map(|(i, t)| b.input(&format!("in{i}"), t.shape().to_vec()))
        .collect::<Result<Vec<_>, _>>()?;
    let out = b.node("out", op, &ids)?;
    let graph = b.finish(&[out])?;
    Ok(backend
        .compile(&graph)?
        .run(inputs)?
        .remove(0)
        .data()
        .to_vec())
}

/// Deterministic finite normal values over several magnitudes, with both zeros.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((state >> 40) as f32) / (1u64 << 24) as f32 - 0.5;
            match i % 11 {
                1 => -0.0,
                2 => 0.0,
                _ => unit * 10f32.powi((state >> 33) as i32 % 7 - 3),
            }
        })
        .collect()
}

fn tensor(shape: &[usize], seed: u64) -> Tensor {
    Tensor::new(shape.to_vec(), values(shape.iter().product(), seed)).unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Every exponent and sign, a handful of mantissas each.
fn pattern_sweep() -> Vec<f32> {
    let mantissas = [0u32, 1, 2, 0x40_0000, 0x55_5555, 0x7f_fffe, 0x7f_ffff];
    let mut out = Vec::new();
    for sign in [0u32, 1] {
        for exponent in 0..=255u32 {
            for m in mantissas {
                out.push(f32::from_bits(sign << 31 | exponent << 23 | m));
            }
        }
    }
    out
}

/// Whether every device result equals its reference result bit for bit.
fn all_exact(pairs: &[(Vec<f32>, Vec<f32>)]) -> bool {
    pairs.iter().all(|(got, want)| bits(got) == bits(want))
}

/// The largest ulp distance of the device Sigmoid from the reference over the finite
/// inputs. A normal reference result is compared in ulps. Below the smallest normal
/// the device may flush to zero, so a subnormal reference only has to pass the flush
/// rule, and a failure there counts as `u32::MAX`.
fn sigmoid_max_ulp(input: &[f32], got: &[f32], want: &[f32]) -> u32 {
    let mut max = 0;
    for i in 0..input.len() {
        if !input[i].is_finite() {
            continue;
        }
        if want[i].is_normal() {
            max = max.max(ulp_distance(got[i], want[i]).unwrap_or(u32::MAX));
        } else if !sigmoid_matches(got[i], want[i], u32::MAX) {
            max = u32::MAX;
        }
    }
    max
}

/// Whether every element of a `(m, k) x (k, n)` product is inside the documented
/// bound of the f64 result, and the largest ulp distance from the reference result.
fn matmul_stats(
    (m, k, n): (usize, usize, usize),
    x: &[f32],
    w: &[f32],
    got: &[f32],
    want: &[f32],
) -> (bool, u32) {
    let (mut within, mut max_ulp) = (true, 0);
    for i in 0..m {
        let row: Vec<f32> = (0..k).map(|t| x[i * k + t]).collect();
        for j in 0..n {
            let col: Vec<f32> = (0..k).map(|t| w[t * n + j]).collect();
            let exact: f64 = row
                .iter()
                .zip(&col)
                .map(|(p, q)| f64::from(*p) * f64::from(*q))
                .sum();
            let idx = i * n + j;
            within &= (f64::from(got[idx]) - exact).abs() <= gpu_matmul_bound(&row, &col);
            max_ulp = max_ulp.max(ulp_distance(got[idx], want[idx]).unwrap_or(u32::MAX));
        }
    }
    (within, max_ulp)
}

/// The sample operands of one MatMul case. The seeds are fixed, so reports compare.
fn matmul_inputs(m: usize, k: usize, n: usize) -> (Tensor, Tensor) {
    (
        tensor(&[m, k], (m * 31 + k) as u64),
        tensor(&[k, n], (n * 17 + k) as u64),
    )
}

/// All cases inside the bound, and the largest ulp distance over all of them.
fn combine_stats(stats: &[(bool, u32)]) -> (bool, u32) {
    (
        stats.iter().all(|(within, _)| *within),
        stats.iter().map(|(_, ulp)| *ulp).max().unwrap_or(0),
    )
}

/// Whether adding zero to the smallest subnormal gave zero, and what adding one to
/// NaN gave.
fn classify_add(subnormal_plus_zero: f32, nan_plus_one: f32) -> (bool, &'static str) {
    let label = if nan_plus_one.is_nan() {
        "nan"
    } else {
        "a number"
    };
    (subnormal_plus_zero == 0.0, label)
}

pub fn run_selftest(ctx: Arc<Context>) -> Result<SelfTestReport, Error> {
    let limits = ctx.limits().clone();
    let info = ctx.info().clone();
    let device = WgpuBackend::for_context(ctx);
    let reference = &ReferenceBackend;

    let a = tensor(&[33, 17], 1);
    let b = tensor(&[33, 17], 2);
    let mut pairs = Vec::new();
    for op in [Op::Add, Op::Mul] {
        let ins = [a.clone(), b.clone()];
        pairs.push((
            run_op(&device, op.clone(), &ins)?,
            run_op(reference, op, &ins)?,
        ));
    }
    for op in [
        Op::Transpose { perm: vec![1, 0] },
        Op::Reshape {
            shape: vec![17, 33],
        },
    ] {
        let ins = [a.clone()];
        pairs.push((
            run_op(&device, op.clone(), &ins)?,
            run_op(reference, op, &ins)?,
        ));
    }
    let finite_tier_exact = all_exact(&pairs);

    let sweep = pattern_sweep();
    let sweep_tensor = Tensor::new(vec![sweep.len()], sweep.clone())?;
    let relu_exact = all_exact(&[(
        run_op(&device, Op::Relu, std::slice::from_ref(&sweep_tensor))?,
        run_op(reference, Op::Relu, std::slice::from_ref(&sweep_tensor))?,
    )]);

    let sigmoid_got = run_op(&device, Op::Sigmoid, std::slice::from_ref(&sweep_tensor))?;
    let sigmoid_want = run_op(reference, Op::Sigmoid, std::slice::from_ref(&sweep_tensor))?;
    let sigmoid_max_ulp = sigmoid_max_ulp(&sweep, &sigmoid_got, &sigmoid_want);

    let mut stats = Vec::new();
    for (m, k, n) in [(1, 1, 1), (17, 16, 15), (33, 33, 33), (64, 64, 64)] {
        let (x, w) = matmul_inputs(m, k, n);
        let got = run_op(&device, Op::MatMul, &[x.clone(), w.clone()])?;
        let want = run_op(reference, Op::MatMul, &[x.clone(), w.clone()])?;
        stats.push(matmul_stats((m, k, n), x.data(), w.data(), &got, &want));
    }
    let (matmul_within_bound, matmul_max_ulp) = combine_stats(&stats);

    let subnormal = Tensor::new(vec![1], vec![f32::from_bits(1)])?;
    let zero = Tensor::new(vec![1], vec![0.0])?;
    let flushed_result = run_op(&device, Op::Add, &[subnormal, zero])?[0];
    let nan = Tensor::new(vec![1], vec![f32::NAN])?;
    let one = Tensor::new(vec![1], vec![1.0])?;
    let nan_result = run_op(&device, Op::Add, &[nan, one])?[0];
    let (flushed, nan_in_add_gives) = classify_add(flushed_result, nan_result);

    Ok(SelfTestReport {
        adapter: info.name,
        api: format!("{:?}", info.backend),
        limits: vec![
            (
                "max_compute_workgroups_per_dimension",
                u64::from(limits.max_compute_workgroups_per_dimension),
            ),
            (
                "max_compute_invocations_per_workgroup",
                u64::from(limits.max_compute_invocations_per_workgroup),
            ),
            (
                "max_compute_workgroup_storage_size",
                u64::from(limits.max_compute_workgroup_storage_size),
            ),
            (
                "max_storage_buffers_per_shader_stage",
                u64::from(limits.max_storage_buffers_per_shader_stage),
            ),
            (
                "max_storage_buffer_binding_size",
                limits.max_storage_buffer_binding_size,
            ),
            ("max_buffer_size", limits.max_buffer_size),
        ],
        finite_tier_exact,
        relu_exact_on_all_sampled_patterns: relu_exact,
        matmul_within_bound,
        matmul_max_ulp,
        sigmoid_max_ulp,
        subnormal_add_flushed: flushed,
        nan_in_add_gives,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // Reports from different versions are compared, so the sample values stay fixed.
    #[test]
    fn the_sample_values_are_pinned() {
        let expected: [u32; 12] = [
            0x38c54999, 0x80000000, 0x00000000, 0x41ec5bb1, 0x36ab8a3d, 0x3d5ceb50, 0xbede815e,
            0x4207e3f4, 0xc040ff96, 0x3980da23, 0x39bb790a, 0x40358eb1,
        ];
        assert_eq!(bits(&values(12, 1)), expected);
        let check = bits(&values(1000, 7))
            .iter()
            .fold(0u32, |a, b| a.wrapping_mul(31).wrapping_add(*b));
        assert_eq!(check, 0xfbbd1938);
    }

    #[test]
    fn the_sample_values_have_the_documented_shape() {
        let v = values(1000, 3);
        assert_eq!(v.len(), 1000);
        assert_eq!(values(0, 3).len(), 0);
        assert_ne!(
            bits(&values(50, 1)),
            bits(&values(50, 2)),
            "the seed matters"
        );
        for (i, x) in v.iter().enumerate() {
            match i % 11 {
                1 => assert_eq!(x.to_bits(), (-0.0f32).to_bits(), "index {i}"),
                2 => assert_eq!(x.to_bits(), 0.0f32.to_bits(), "index {i}"),
                _ => assert!(x.is_finite() && x.abs() < 600.0, "index {i}: {x}"),
            }
        }
        let big = v.iter().filter(|x| x.abs() > 10.0).count();
        let small = v.iter().filter(|x| **x != 0.0 && x.abs() < 0.01).count();
        assert!(
            big > 20 && small > 20,
            "values span several magnitudes: {big} {small}"
        );
    }

    #[test]
    fn bits_reads_the_raw_pattern() {
        assert_eq!(
            bits(&[1.0, -0.0, f32::INFINITY]),
            [0x3f80_0000, 0x8000_0000, 0x7f80_0000]
        );
        assert!(bits(&[]).is_empty());
    }

    #[test]
    fn the_pattern_sweep_has_every_exponent_sign_and_mantissa_once() {
        let sweep = pattern_sweep();
        assert_eq!(sweep.len(), 2 * 256 * 7);
        let all: HashSet<u32> = bits(&sweep).into_iter().collect();
        assert_eq!(all.len(), sweep.len(), "no pattern repeats");
        for sign in [0u32, 1] {
            for exponent in 0..=255u32 {
                for m in [0u32, 1, 2, 0x40_0000, 0x55_5555, 0x7f_fffe, 0x7f_ffff] {
                    assert!(all.contains(&(sign << 31 | exponent << 23 | m)));
                }
            }
        }
        assert_eq!(sweep[0].to_bits(), 0);
        assert_eq!(sweep[sweep.len() - 1].to_bits(), 0xffff_ffff);
    }

    #[test]
    fn all_exact_needs_every_pair_equal_bit_for_bit() {
        let same = (vec![1.0, -0.0], vec![1.0, -0.0]);
        assert!(all_exact(&[]));
        assert!(all_exact(&[same.clone(), same.clone()]));
        assert!(!all_exact(&[
            same.clone(),
            (vec![1.0], vec![1.0f32.next_up()])
        ]));
        assert!(
            !all_exact(&[(vec![0.0], vec![-0.0]), same.clone()]),
            "signed zeros differ"
        );
        assert!(!all_exact(&[(vec![1.0], vec![1.0, 2.0])]), "lengths differ");
    }

    #[test]
    fn sigmoid_ulp_is_the_largest_distance_over_finite_inputs() {
        let input = [0.5f32, -3.0, 2.0];
        let want = [0.6224593f32, 0.04742587, 0.8807971];
        assert_eq!(sigmoid_max_ulp(&input, &want, &want), 0);
        let mut got = want;
        got[1] = want[1].next_up();
        assert_eq!(sigmoid_max_ulp(&input, &got, &want), 1);
        got[2] = want[2].next_up().next_up().next_up();
        assert_eq!(sigmoid_max_ulp(&input, &got, &want), 3);
        got[0] = f32::NAN;
        assert_eq!(sigmoid_max_ulp(&input, &got, &want), u32::MAX);
    }

    #[test]
    fn sigmoid_ulp_ignores_non_finite_inputs_and_allows_a_flush_below_the_smallest_normal() {
        let tiny = f32::MIN_POSITIVE / 4.0;
        // A device that flushes a subnormal reference to zero is fine.
        assert_eq!(sigmoid_max_ulp(&[-100.0], &[0.0], &[tiny]), 0);
        // A subnormal reference that the device gets far wrong is not.
        assert_eq!(sigmoid_max_ulp(&[-100.0], &[1e-30], &[tiny]), u32::MAX);
        // A non-finite input is skipped whatever the results are.
        assert_eq!(
            sigmoid_max_ulp(&[f32::NAN, f32::INFINITY], &[9.0, 9.0], &[0.5, 1.0]),
            0
        );
        // A normal reference with a flushed result is a failure, not a flush.
        assert_eq!(
            sigmoid_max_ulp(&[0.0], &[0.0], &[0.5]),
            ulp_distance(0.0, 0.5).unwrap()
        );
    }

    // A 2 x 3 times 3 x 4 product, with distinct values so a wrong index changes the sums.
    fn small_product() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let x: Vec<f32> = (1..=6).map(|v| v as f32 * 0.5).collect();
        let w: Vec<f32> = (1..=12).map(|v| v as f32 * 0.25 - 1.0).collect();
        let mut out = vec![0.0f32; 8];
        for i in 0..2 {
            for j in 0..4 {
                let sum: f64 = (0..3)
                    .map(|t| f64::from(x[i * 3 + t]) * f64::from(w[t * 4 + j]))
                    .sum();
                out[i * 4 + j] = sum as f32;
            }
        }
        (x, w, out)
    }

    #[test]
    fn matmul_stats_accepts_the_exact_product_with_zero_ulp() {
        let (x, w, out) = small_product();
        assert_eq!(matmul_stats((2, 3, 4), &x, &w, &out, &out), (true, 0));
    }

    #[test]
    fn matmul_stats_counts_the_ulp_difference_from_the_reference_result() {
        let (x, w, out) = small_product();
        let mut got = out.clone();
        got[5] = out[5].next_up();
        assert_eq!(matmul_stats((2, 3, 4), &x, &w, &got, &out), (true, 1));
        got[7] = out[7].next_up().next_up();
        assert_eq!(matmul_stats((2, 3, 4), &x, &w, &got, &out), (true, 2));
        got[0] = f32::NAN;
        assert_eq!(matmul_stats((2, 3, 4), &x, &w, &got, &out).1, u32::MAX);
    }

    #[test]
    fn matmul_stats_rejects_a_result_outside_the_bound_in_any_element() {
        let (x, w, out) = small_product();
        for idx in 0..8 {
            let mut got = out.clone();
            got[idx] += 1.0;
            assert!(
                !matmul_stats((2, 3, 4), &x, &w, &got, &out).0,
                "element {idx}"
            );
        }
        // Swapping two elements is also wrong: the index arithmetic matters.
        let mut swapped = out.clone();
        swapped.swap(1, 6);
        assert!(!matmul_stats((2, 3, 4), &x, &w, &swapped, &out).0);
    }

    #[test]
    fn classify_add_reads_the_two_probe_results() {
        assert_eq!(classify_add(0.0, f32::NAN), (true, "nan"));
        assert_eq!(classify_add(-0.0, 1.0), (true, "a number"));
        assert_eq!(classify_add(f32::from_bits(1), f32::NAN), (false, "nan"));
        assert_eq!(classify_add(f32::from_bits(1), 2.0), (false, "a number"));
    }

    #[test]
    fn the_matmul_sample_operands_use_fixed_seeds() {
        let (x, w) = matmul_inputs(17, 16, 15);
        assert_eq!(x, tensor(&[17, 16], 543));
        assert_eq!(w, tensor(&[16, 15], 271));
        let (x, w) = matmul_inputs(2, 3, 4);
        assert_eq!(x, tensor(&[2, 3], 65));
        assert_eq!(w, tensor(&[3, 4], 71));
    }

    #[test]
    fn combining_cases_needs_all_inside_the_bound_and_takes_the_largest_ulp() {
        assert_eq!(combine_stats(&[]), (true, 0));
        assert_eq!(combine_stats(&[(true, 3), (true, 9), (true, 1)]), (true, 9));
        assert_eq!(combine_stats(&[(true, 3), (false, 1)]), (false, 3));
        assert_eq!(combine_stats(&[(false, 0)]), (false, 0));
    }
}
