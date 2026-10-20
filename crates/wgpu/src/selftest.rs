//! A measurement of one device against the reference backend, in the same terms
//! as the contract: what is exact, what is bounded, what the device does with
//! the values the shading language does not pin down. Anyone with another
//! device can run it (`oxfuse gpu-selftest`) and attach the output to an issue.

use crate::bounds::{SIGMOID_ULP_BOUND, matmul_bound, sigmoid_matches, ulp_distance};
use crate::config::KernelConfig;
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

pub fn run_selftest(ctx: Arc<Context>) -> Result<SelfTestReport, Error> {
    let limits = ctx.limits().clone();
    let info = ctx.info().clone();
    let device = WgpuBackend::with(ctx, KernelConfig::DEFAULT);
    let reference = &ReferenceBackend;

    let mut finite_tier_exact = true;
    let a = tensor(&[33, 17], 1);
    let b = tensor(&[33, 17], 2);
    for op in [Op::Add, Op::Mul] {
        let ins = [a.clone(), b.clone()];
        finite_tier_exact &=
            bits(&run_op(&device, op.clone(), &ins)?) == bits(&run_op(reference, op, &ins)?);
    }
    for op in [
        Op::Transpose { perm: vec![1, 0] },
        Op::Reshape {
            shape: vec![17, 33],
        },
    ] {
        let ins = [a.clone()];
        finite_tier_exact &=
            bits(&run_op(&device, op.clone(), &ins)?) == bits(&run_op(reference, op, &ins)?);
    }

    let sweep = pattern_sweep();
    let sweep_tensor = Tensor::new(vec![sweep.len()], sweep.clone())?;
    let relu_exact = bits(&run_op(
        &device,
        Op::Relu,
        std::slice::from_ref(&sweep_tensor),
    )?) == bits(&run_op(
        reference,
        Op::Relu,
        std::slice::from_ref(&sweep_tensor),
    )?);

    let sigmoid_got = run_op(&device, Op::Sigmoid, std::slice::from_ref(&sweep_tensor))?;
    let sigmoid_want = run_op(reference, Op::Sigmoid, std::slice::from_ref(&sweep_tensor))?;
    let mut sigmoid_max_ulp = 0;
    for i in 0..sweep.len() {
        if sweep[i].is_finite() && sigmoid_want[i].is_normal() {
            let ulp = ulp_distance(sigmoid_got[i], sigmoid_want[i]);
            sigmoid_max_ulp = sigmoid_max_ulp.max(ulp.unwrap_or(u32::MAX));
        } else if sweep[i].is_finite()
            && !sigmoid_matches(sigmoid_got[i], sigmoid_want[i], u32::MAX)
        {
            sigmoid_max_ulp = u32::MAX;
        }
    }

    let (mut matmul_within_bound, mut matmul_max_ulp) = (true, 0);
    for (m, k, n) in [(1, 1, 1), (17, 16, 15), (33, 33, 33), (64, 64, 64)] {
        let x = tensor(&[m, k], (m * 31 + k) as u64);
        let w = tensor(&[k, n], (n * 17 + k) as u64);
        let got = run_op(&device, Op::MatMul, &[x.clone(), w.clone()])?;
        let want = run_op(reference, Op::MatMul, &[x.clone(), w.clone()])?;
        for i in 0..m {
            let row: Vec<f32> = (0..k).map(|t| x.data()[i * k + t]).collect();
            for j in 0..n {
                let col: Vec<f32> = (0..k).map(|t| w.data()[t * n + j]).collect();
                let exact: f64 = row
                    .iter()
                    .zip(&col)
                    .map(|(p, q)| f64::from(*p) * f64::from(*q))
                    .sum();
                let idx = i * n + j;
                matmul_within_bound &=
                    (f64::from(got[idx]) - exact).abs() <= matmul_bound(&row, &col);
                matmul_max_ulp =
                    matmul_max_ulp.max(ulp_distance(got[idx], want[idx]).unwrap_or(u32::MAX));
            }
        }
    }

    let subnormal = Tensor::new(vec![1], vec![f32::from_bits(1)])?;
    let zero = Tensor::new(vec![1], vec![0.0])?;
    let flushed = run_op(&device, Op::Add, &[subnormal, zero])?[0] == 0.0;
    let nan = Tensor::new(vec![1], vec![f32::NAN])?;
    let one = Tensor::new(vec![1], vec![1.0])?;
    let nan_in_add_gives = if run_op(&device, Op::Add, &[nan, one])?[0].is_nan() {
        "nan"
    } else {
        "a number"
    };

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
